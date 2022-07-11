import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import json
from typing import Optional, Any
from const import ROOT_DIR
from scipy.special import softmax
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
)

from transformers import (
    AutoTokenizer,
    AutoConfig,
    Trainer,
    EvalPrediction
)

from multimodal_transformers.data import load_data
from multimodal_transformers.model import TabularConfig
from multimodal_transformers.model import AutoModelWithTabular

@dataclass
class ModelArguments:
  """
  Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
  """

  model_name_or_path: str = field(
      metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
  )
  config_name: Optional[str] = field(
      default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
  )
  tokenizer_name: Optional[str] = field(
      default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
  )
  cache_dir: Optional[str] = field(
      default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
  )

@dataclass
class MultimodalDataTrainingArguments:
  """
  Arguments pertaining to how we combine tabular features
  Using `HfArgumentParser` we can turn this class
  into argparse arguments to be able to specify them on
  the command line.
  """

  data_path: Optional[str] = field(default=None, metadata={
                            'help': 'the path to the csv file containing the dataset'
                        })
  column_info_path: str = field(
      default=None,
      metadata={
          'help': 'the path to the json file detailing which columns are text, categorical, numerical, and the label'
  })

  column_info: dict = field(
      default=None,
      metadata={
          'help': 'a dict referencing the text, categorical, numerical, and label columns'
                  'its keys are text_cols, num_cols, cat_cols, and label_col'
  })

  categorical_encode_type: Optional[str] = field(default=None,
                                        metadata={
                                            'help': 'sklearn encoder to use for categorical data',
                                            'choices': ['ohe', 'binary', 'label', 'none']
                                        })
  numerical_transformer_method: str = field(default='yeo_johnson',
                                            metadata={
                                                'help': 'sklearn numerical transformer to preprocess numerical data',
                                                'choices': ['yeo_johnson', 'box_cox', 'quantile_normal', 'none']
                                            })
  task: str = field(default="classification",
                    metadata={
                        "help": "The downstream training task",
                        "choices": ["classification", "regression"]
                    })

  mlp_division: int = field(default=4,
                            metadata={
                                'help': 'the ratio of the number of '
                                        'hidden dims in a current layer to the next MLP layer'
                            })
  combine_feat_method: str = field(default='individual_mlps_on_cat_and_numerical_feats_then_concat',
                                    metadata={
                                        'help': 'method to combine categorical and numerical features '
                                    })
  mlp_dropout: float = field(default=0.1,
                              metadata={
                                'help': 'dropout ratio used for MLP layers'
                              })
  numerical_bn: bool = field(default=True,
                              metadata={
                                  'help': 'whether to use batchnorm on numerical features'
                              })
  use_simple_classifier: str = field(default=True,
                                      metadata={
                                          'help': 'whether to use single layer or MLP as final classifier'
                                      })
  mlp_act: str = field(default='relu',
                        metadata={
                            'help': 'the activation function to use for finetuning layers',
                            'choices': ['relu', 'prelu', 'sigmoid', 'tanh', 'linear']
                        })
  gating_beta: float = field(default=0.2,
                              metadata={
                                  'help': "the beta hyperparameters used for gating tabular data "
                                          "see https://www.aclweb.org/anthology/2020.acl-main.214.pdf"
                              })

  def __post_init__(self):
      assert self.column_info != self.column_info_path
      if self.column_info is None and self.column_info_path:
          with open(self.column_info_path, 'r') as f:
              self.column_info = json.load(f)

class MixedModelClassification():
    def __init__(self, model_args: dataclass, data_args: dataclass, num_labels: int, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> None:
        
        self.version = 1
        self.config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            cache_dir = model_args.cache_dir,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
                model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
        )

        self.dataframes = (self.convert_dataset(train, data_args), self.convert_dataset(validation, data_args), self.convert_dataset(test, data_args))


        self.config.tabular_config = TabularConfig(
                num_labels = num_labels,
                cat_feat_dim = self.dataframes[0].cat_feats.shape[1],
                numerical_feat_dim = self.dataframes[0].numerical_feats.shape[1],
                **vars(data_args))

        self.model = AutoModelWithTabular.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            config = self.config,
            cache_dir = model_args.cache_dir
        )

        self.trainer = None
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:

       return self.model(*args, **kwds)


    def convert_dataset(self, data: pd.DataFrame, data_args: dataclass) -> 'tabular_torch_dataset.TorchTextDataset':
        """AI is creating summary for convert_dataset

        Args:
            data (pd.DataFrame): dataframe to convert

        Returns:
            TorchTextDataset: converted Dataset
        """
        converted = load_data(
            data,
            text_cols = data_args.column_info['text_cols'],
            tokenizer = self.tokenizer,
            label_col = data_args.column_info['label_col'],
            label_list = data_args.column_info['label_list'],
            categorical_cols = data_args.column_info['cat_cols'],
            numerical_cols = data_args.column_info['num_cols'],
            sep_text_token_str = self.tokenizer.sep_token,
        )
        return converted

    @staticmethod
    def calc_classification_metrics(p: EvalPrediction):
        pred_labels = np.argmax(p.predictions, axis=1)
        pred_scores = softmax(p.predictions, axis=1)[:, 1]
        labels = p.label_ids
        if len(np.unique(labels)) == 2:  # binary classification
            roc_auc_pred_score = roc_auc_score(labels, pred_scores)
            precisions, recalls, thresholds = precision_recall_curve(labels,
                                                                        pred_scores)
            fscore = (2 * precisions * recalls) / (precisions + recalls)
            fscore[np.isnan(fscore)] = 0
            ix = np.argmax(fscore)
            threshold = thresholds[ix].item()
            pr_auc = auc(recalls, precisions)
            tn, fp, fn, tp = confusion_matrix(labels, pred_labels, labels=[0, 1]).ravel()
            result = {'roc_auc': roc_auc_pred_score,
                        'threshold': threshold,
                        'pr_auc': pr_auc,
                        'recall': recalls[ix].item(),
                        'precision': precisions[ix].item(), 'f1': fscore[ix].item(),
                        'tn': tn.item(), 'fp': fp.item(), 'fn': fn.item(), 'tp': tp.item()
                        }
        else:
            acc = (pred_labels == labels).mean()
            f1 = f1_score(y_true=labels, y_pred=pred_labels)
            result = {
                "acc": acc,
                "f1": f1,
                "acc_and_f1": (acc + f1) / 2,
                "mcc": matthews_corrcoef(labels, pred_labels)
            }

        return result 
    
    def train(self, training_args: dataclass):
        self.trainer = Trainer(
        model = self.model,
        args = training_args,
        train_dataset = self.dataframes[0], #train
        eval_dataset = self.dataframes[1], #evaluation
        compute_metrics = self.calc_classification_metrics,
        )
    
        self.trainer.train()
        self.save()

    def predict(self, data) -> None:
        if self.trainer is None:
            raise ValueError("fisrt train the model")
        
        return self.trainer.predict(data)
    
    def save(self) -> None:

        model_dir = os.path.join(ROOT_DIR, f"files/models/MixedMultiModal_v{self.version}")
        os.mkdir(model_dir)
        self.trainer.save_model(model_dir)