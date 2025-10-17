import inspect
import logging
import os
import warnings
from typing import Optional, Tuple

import cloudpickle
import torch
import torch.nn.functional as F

from ...customexception import BadInheritanceError
from ...data.datasets import BaseDataset
from ..base import BaseAE
from ..base.base_utils import CPU_Unpickler, ModelOutput, hf_hub_is_available
from ..nn import BaseDecoder, BaseEncoder
from .amortized_dual_vae_config import AmortizedDualVAEConfig
from .langevin import LangevinPCD
from .monomials import MonomialBasis
from .networks import BaseLambdaNet, LambdaNetMLP, MomentsEncoderMLP

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class AmortizedDualVAE(BaseAE):
    """VAE with an energy-based latent prior and amortised dual solver."""

    def __init__(
        self,
        model_config: AmortizedDualVAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
        lambda_net: Optional[BaseLambdaNet] = None,
        sampler: Optional[LangevinPCD] = None,
    ):
        BaseAE.__init__(self, model_config=model_config, decoder=decoder)

        self.model_name = "AmortizedDualVAE"

        if encoder is None:
            encoder = MomentsEncoderMLP(model_config)
            self.model_config.uses_default_encoder = True
        else:
            self.model_config.uses_default_encoder = False

        self.set_encoder(encoder)

        if lambda_net is None:
            lambda_net = LambdaNetMLP(model_config)
            self.model_config.uses_default_lambda_net = True
        else:
            self.model_config.uses_default_lambda_net = False

        self.set_lambda_net(lambda_net)

        self.basis = MonomialBasis(
            latent_dim=model_config.latent_dim,
            order=model_config.polynomial_order,
            exclude_constant=True,
        )

        self.sampler = sampler or LangevinPCD(
            latent_dim=model_config.latent_dim,
            steps=model_config.langevin_steps,
            step_size=model_config.langevin_step_size,
            n_samples=model_config.langevin_n_samples,
            c=model_config.energy_scale,
            reinit_prob=model_config.langevin_reinit_prob,
            noise_scale=model_config.langevin_noise_scale,
            update_clamp=model_config.langevin_update_clamp,
        )

    def set_lambda_net(self, lambda_net: BaseLambdaNet) -> None:
        if not issubclass(type(lambda_net), BaseLambdaNet):
            raise BadInheritanceError(
                (
                    "Lambda network must inherit from `BaseLambdaNet`. "
                    "Refer to documentation for custom architectures."
                )
            )

        self.lambda_net = lambda_net

    def forward(self, inputs: BaseDataset, **kwargs) -> ModelOutput:
        x = inputs["data"]

        encoder_output = self.encoder(x)
        moment_hat = encoder_output.embedding

        lam = self.lambda_net(moment_hat)

        z_samples = self.sampler.sample(lam.detach(), self.basis)
        batch_size, num_samples, latent_dim = z_samples.shape

        flat_z = z_samples.reshape(batch_size * num_samples, latent_dim)
        decoder_output = self.decoder(flat_z)
        recon = decoder_output["reconstruction"].reshape(
            batch_size, num_samples, *self.input_dim
        )
        recon_mean = recon.mean(dim=1)

        rec_loss, rec_matrix = self._reconstruction_terms(recon, x)

        features = self.basis(z_samples.reshape(-1, latent_dim)).reshape(
            batch_size, num_samples, -1
        )
        feature_mean = features.mean(dim=1)

        centered_rec = (rec_matrix - rec_matrix.mean(dim=1, keepdim=True)).detach()
        centered_features = (features - feature_mean.unsqueeze(1)).detach()

        score_proxy = (
            centered_rec.unsqueeze(-1) * centered_features * lam.unsqueeze(1)
        ).mean()

        dual_proxy = ((feature_mean.detach() - moment_hat) * lam).sum(dim=-1).mean()

        moment_loss = F.mse_loss(feature_mean.detach(), moment_hat)

        lambda_reg = (lam**2).mean()
        moment_reg = (moment_hat**2).mean()

        total_loss = (
            rec_loss
            + self.model_config.score_weight * score_proxy
            + self.model_config.dual_weight * dual_proxy
            + self.model_config.moment_weight * moment_loss
            + self.model_config.lambda_reg_weight * lambda_reg
        )
        print("reconstruction loss:", rec_loss.item(),
              "score proxy:", score_proxy.item(),
              "dual proxy:", dual_proxy.item(),
              "moment loss:", moment_loss.item(),
              "lambda reg:", lambda_reg.item())

        if self.model_config.moment_reg_weight > 0:
            total_loss = total_loss + self.model_config.moment_reg_weight * moment_reg

        output = ModelOutput(
            loss=total_loss,
            recon_loss=rec_loss,
            score_proxy=score_proxy,
            dual_proxy=dual_proxy,
            moment_loss=moment_loss,
            lambda_reg=lambda_reg,
            moment_reg=moment_reg,
            recon_x=recon_mean,
            z=z_samples[:, -1, :],
            lambda_params=lam,
            moment_target=moment_hat,
            feature_expectation=feature_mean,
        )

        return output

    def _reconstruction_terms(
        self, recon: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_samples = recon.shape[0], recon.shape[1]
        recon_flat = recon.reshape(batch_size * num_samples, -1)
        target = x.unsqueeze(1).expand_as(recon)
        target_flat = target.reshape(batch_size * num_samples, -1)

        if self.model_config.reconstruction_loss == "mse":
            per_sample = 0.5 * F.mse_loss(
                recon_flat, target_flat, reduction="none"
            ).sum(dim=-1)
        else:
            per_sample = F.binary_cross_entropy(
                recon_flat, target_flat, reduction="none"
            ).sum(dim=-1)

        per_sample = per_sample.reshape(batch_size, num_samples)
        return per_sample.mean(), per_sample

    def save(self, dir_path: str):
        super().save(dir_path)

        if not self.model_config.uses_default_lambda_net:
            with open(os.path.join(dir_path, "lambda_net.pkl"), "wb") as fp:
                cloudpickle.register_pickle_by_value(inspect.getmodule(self.lambda_net))
                cloudpickle.dump(self.lambda_net, fp)

    @classmethod
    def _load_custom_lambda_net_from_folder(cls, dir_path: str) -> BaseLambdaNet:
        file_list = os.listdir(dir_path)
        cls._check_python_version_from_folder(dir_path=dir_path)

        if "lambda_net.pkl" not in file_list:
            raise FileNotFoundError(
                f"Missing lambda network file ('lambda_net.pkl') in {dir_path}."
            )

        with open(os.path.join(dir_path, "lambda_net.pkl"), "rb") as fp:
            lambda_net = CPU_Unpickler(fp).load()

        return lambda_net

    @classmethod
    def load_from_folder(cls, dir_path):
        model_config = cls._load_model_config_from_folder(dir_path)
        model_weights = cls._load_model_weights_from_folder(dir_path)

        if not model_config.uses_default_encoder:
            encoder = cls._load_custom_encoder_from_folder(dir_path)
        else:
            encoder = None

        if not model_config.uses_default_decoder:
            decoder = cls._load_custom_decoder_from_folder(dir_path)
        else:
            decoder = None

        if not model_config.uses_default_lambda_net:
            lambda_net = cls._load_custom_lambda_net_from_folder(dir_path)
        else:
            lambda_net = None

        model = cls(
            model_config,
            encoder=encoder,
            decoder=decoder,
            lambda_net=lambda_net,
        )
        model.load_state_dict(model_weights)

        return model

    @classmethod
    def load_from_hf_hub(cls, hf_hub_path: str, allow_pickle=False):
        if not hf_hub_is_available():
            raise ModuleNotFoundError(
                "`huggingface_hub` package must be installed to load models from the HF hub. "
                "Run `python -m pip install huggingface_hub` and log in to your account with "
                "`huggingface-cli login`."
            )

        from huggingface_hub import hf_hub_download

        logger.info(f"Downloading {cls.__name__} files for rebuilding...")

        _ = hf_hub_download(repo_id=hf_hub_path, filename="environment.json")
        config_path = hf_hub_download(repo_id=hf_hub_path, filename="model_config.json")
        dir_path = os.path.dirname(config_path)

        _ = hf_hub_download(repo_id=hf_hub_path, filename="model.pt")

        model_config = cls._load_model_config_from_folder(dir_path)

        if (
            cls.__name__ + "Config" != model_config.name
            and cls.__name__ + "_Config" != model_config.name
        ):
            warnings.warn(
                f"You are trying to load a `{cls.__name__}` while a `{model_config.name}` is given."
            )

        model_weights = cls._load_model_weights_from_folder(dir_path)

        if (
            not model_config.uses_default_encoder
            or not model_config.uses_default_decoder
            or not model_config.uses_default_lambda_net
        ) and not allow_pickle:
            warnings.warn(
                "You are about to download pickled files from the HF hub that may have been "
                "created by a third party and so could potentially harm your computer. If you are "
                "sure that you want to download them set `allow_pickle=True`."
            )

        if not model_config.uses_default_encoder:
            _ = hf_hub_download(repo_id=hf_hub_path, filename="encoder.pkl")
            encoder = cls._load_custom_encoder_from_folder(dir_path)
        else:
            encoder = None

        if not model_config.uses_default_decoder:
            _ = hf_hub_download(repo_id=hf_hub_path, filename="decoder.pkl")
            decoder = cls._load_custom_decoder_from_folder(dir_path)
        else:
            decoder = None

        if not model_config.uses_default_lambda_net:
            _ = hf_hub_download(repo_id=hf_hub_path, filename="lambda_net.pkl")
            lambda_net = cls._load_custom_lambda_net_from_folder(dir_path)
        else:
            lambda_net = None

        logger.info(f"Successfully downloaded {cls.__name__} model!")

        model = cls(
            model_config,
            encoder=encoder,
            decoder=decoder,
            lambda_net=lambda_net,
        )
        model.load_state_dict(model_weights)

        return model
