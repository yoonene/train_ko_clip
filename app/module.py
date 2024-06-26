import inspect
from itertools import chain
from typing import Literal

import pytorch_lightning as pl
import torch
from loguru import logger
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
)

from .util import create_optimizer, refresh_access_token, upload_folder_to_google_drive


class KoCLIPModule(pl.LightningModule):
    def __init__(
        self,
        teacher_model_name: str,
        student_model_name: str,
        model_type: Literal["clip", "dual_encoder"] = "clip",
        optimizer: str = "adamw",
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
        use_auth_token: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        # init model
        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name
        self.model_type = model_type
        self.use_auth_token = use_auth_token
        self.teacher, self.student = self.init_model(
            teacher_model_name, student_model_name
        )

        self.mse = torch.nn.MSELoss()
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.refresh_token = ""
        self.client_id = ""
        self.client_secret = ""

    def init_model(self, teacher_model_name: str, student_model_name: str):
        teacher = CLIPModel.from_pretrained(
            teacher_model_name, use_auth_token=self.use_auth_token
        )

        if self.model_type == "clip":
            student = CLIPModel.from_pretrained(
                student_model_name, use_auth_token=self.use_auth_token
            )
        else:
            student = VisionTextDualEncoderModel.from_vision_text_pretrained(
                teacher_model_name, student_model_name
            )

            vp_state = teacher.visual_projection.state_dict()
            student.visual_projection.load_state_dict(vp_state)
            student.logit_scale = teacher.logit_scale

        return teacher, student

    def configure_optimizers(self):
        if self.model_type == "clip":
            params = list(self.student.text_model.named_parameters())
        else:
            params = list(
                chain(
                    self.student.text_model.named_parameters(),
                    self.student.text_projection.named_parameters(),
                )
            )

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        opt_class = create_optimizer(self.optimizer)
        signiture = inspect.signature(opt_class)
        opt_kwargs = {}
        if "capturable" in signiture.parameters:
            opt_kwargs["capturable"] = True
        if "weight_decouple" in signiture.parameters:
            opt_kwargs["weight_decouple"] = True
        if "decouple_decay" in signiture.parameters:
            opt_kwargs["decouple_decay"] = True

        optimizer = opt_class(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            **opt_kwargs,
        )

        if "bnb" in self.optimizer:
            from bitsandbytes.optim import GlobalOptimManager

            manager = GlobalOptimManager.get_instance()
            modules = chain(
                self.student.text_model.modules(),
                self.student.text_projection.modules(),
            )
            for module in modules:
                if isinstance(module, torch.nn.Embedding):
                    manager.register_module_override(
                        module, "weight", {"optim_bits": 32}
                    )
                    logger.debug(f"bitsandbytes: will optimize {module} in fp32")

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
        )

        scheduler_config = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler_config]

    def step(self, batch):
        ko_batch, en_ko_batch, en_en_batch = batch
        # logger.info(f"ko_batch shape: {ko_batch['input_ids'].size()}")
        # logger.info(f"en_ko_batch shape: {en_ko_batch['input_ids'].size()}")
        # logger.info(f"en_en_batch shape: {en_en_batch['input_ids'].size()}")

        if self.model_type == "clip":
            logger.info(f"**ko_batch: {ko_batch}")
            # ko_emb = self.student.text_model(**ko_batch).last_hidden_state
            ko_emb = self.student.text_model(**ko_batch)[1]
            # logger.info(f"ko_emb: {ko_emb}")
            # en_ko_emb = self.student.text_model(**en_ko_batch).last_hidden_state
            en_ko_emb = self.student.text_model(**en_ko_batch)[1]
            # logger.info(f"en_ko_emb: {en_ko_emb}")
            # en_en_emb = self.teacher.text_model(**en_en_batch).last_hidden_state
            en_en_emb = self.teacher.text_model(**en_en_batch)[1]
            # logger.info(f"en_en_emb: {en_en_emb}")
        else:
            ko_emb = self.student.get_text_features(**ko_batch)
            en_ko_emb = self.student.get_text_features(**en_ko_batch)
            en_en_emb = self.teacher.get_text_features(**en_en_batch)

        # logger.info(f"ko_emb shape: {ko_emb.size()}")
        # loggerl.info(f"en_ko_emb shape: {en_ko_emb.size()}")
        # logger.info(f"en_en_emb shape: {en_en_emb.size()}")

        ko_en_loss = self.mse(
            ko_emb, en_en_emb
        )  # 한국어 텍스트에 대한 teacher model의 지식을 반영, student model이 한국어 텍스트에서 teacher model이 영어 텍스트에서 추출한 특성(특히 의미적 특징)을 잘 학습하고 있는지 평가
        en_en_loss = self.mse(
            en_ko_emb, en_en_emb
        )  # 영어 텍스트에 대해 일관된 특성을 학습, student model이 영어 텍스트에서 teacher model이 영어 텍스트에서 추출한 특성을 잘 학습하고 있는지 평가
        loss = ko_en_loss + en_en_loss

        loss_dict = {
            "loss": loss,
            "loss_ko": ko_en_loss,
            "loss_en": en_en_loss,
        }

        return loss_dict

    def training_step(self, batch, batch_idx):
        # logger.info(f"step_batch: {batch}")
        loss = self.step(batch)

        self.log_dict(
            {
                "train/loss": loss["loss"],
                "train/loss_ko": loss["loss_ko"],
                "train/loss_en": loss["loss_en"],
            },
            on_step=True,
            on_epoch=True,
        )
        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)

        self.log_dict(
            {
                "val/loss": loss["loss"],
                "val/loss_ko": loss["loss_ko"],
                "val/loss_en": loss["loss_en"],
            },
            on_epoch=True,
        )
        return loss["loss"]

    def save(self, save_dir: str = "save/my_model"):

        self.student.save_pretrained(save_dir)

        tokenizer = AutoTokenizer.from_pretrained(
            self.student_model_name, use_auth_token=self.use_auth_token
        )

        if self.model_type == "clip":
            processor = CLIPProcessor.from_pretrained(
                self.student_model_name, use_auth_token=self.use_auth_token
            )
        else:
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.teacher_model_name, use_auth_token=self.use_auth_token
            )
            processor = VisionTextDualEncoderProcessor(feature_extractor, tokenizer)
        processor.save_pretrained(save_dir)

        # refresh access token (Colab 사용시 활성화)
        access_token = refresh_access_token(
            self.client_id, self.client_secret, self.refresh_token
        )
        # Upload the saved directory to Google Drive
        upload_folder_to_google_drive(access_token, folder_path=save_dir)
