import pandas as pd
import numpy as np
import base
import os

from sklearn.preprocessing import Binarizer, LabelEncoder, StandardScaler
from model import ModelArguments, MultimodalDataTrainingArguments, MixedModelClassification
from transformers.training_args import TrainingArguments
from transformers import set_seed
from base import Base
from const import ROOT_DIR

general_path = os.path.join(ROOT_DIR, "files/bases/general.csv")
fines_path = os.path.join(ROOT_DIR, "files/bases/multas.csv")

columns_categorical = {
    'nivel_entidad': Base.convert_to_preprocess_text(LabelEncoder),
    'orden_entidad': Base.convert_to_preprocess_text(LabelEncoder),
    'tipo_de_proceso': Base.convert_to_preprocess_text(LabelEncoder),
    'estado_del_proceso': Base.convert_to_preprocess_text(LabelEncoder),
    'otras_formas_contratacion': Base.convert_to_preprocess_text(LabelEncoder),
    'regimen_de_contratacion': Base.convert_to_preprocess_text(LabelEncoder),
    'objeto_a_contratar': Base.convert_to_preprocess_text(LabelEncoder),
    'tipo_de_contrato': Base.convert_to_preprocess_text(LabelEncoder),
    'municipio_obtencion': Base.convert_to_preprocess_text(LabelEncoder),
    'municipio_entrega': Base.convert_to_preprocess_text(LabelEncoder),
    'municipios_ejecucion': Base.convert_to_preprocess_text(LabelEncoder),
    'nombre_grupo': Base.convert_to_preprocess_text(LabelEncoder),
    'nombre_familia': Base.convert_to_preprocess_text(LabelEncoder),
    'nombre_clase': Base.convert_to_preprocess_text(LabelEncoder),
    'origen_de_los_recursos': Base.convert_to_preprocess_text(LabelEncoder),
    'calificacion_definitiva': Base.convert_to_preprocess_text(LabelEncoder),
    'moneda': Base.convert_to_preprocess_text(LabelEncoder),
    'marcacion_adiciones': Base.convert_to_preprocess_text(Binarizer),
    'nombre_rubro': Base.convert_to_preprocess_text(LabelEncoder),
    'municipio_entidad': Base.convert_to_preprocess_text(LabelEncoder),
    'departamento_entidad': Base.convert_to_preprocess_text(LabelEncoder),
}

columns_numerical = [
    'cuantia_proceso',
    'plazo_de_ejec_del_contrato',
    'tiempo_adiciones_en_dias',
    'tiempo_adiciones_en_meses',
    'valor_contrato_con_adiciones'
]

data = base.Base.load_data([
    fines_path,
    general_path
])  

if data.exist_temp is False:
    data.save()

#transform the categorical
for column, func in columns_categorical.items():
    data.enconders[column], data.base[column] = func(data[column])

#transform the numerical
data.stadarize_numbers(columns=columns_numerical, scaler = StandardScaler)

#transform the text
data.preprocess_text(column = "detalle_del_objeto_a_contratar")

# multi mixed model layer 
column_info_dict = {
    'text_cols': [col for col in data.base.columns if col not in [*columns_numerical, *columns_numerical]],
    'num_cols': columns_numerical,
    'cat_cols': list(columns_categorical.keys()),
    'label_col': data.label,
    'label_list': list(data[data.label].unique())
}

model_args = ModelArguments(
    model_name_or_path='dccuchile/bert-base-spanish-wwm-uncased'
)

data_args = MultimodalDataTrainingArguments(
    combine_feat_method='gating_on_cat_and_num_feats_then_sum',
    column_info=column_info_dict,
    task='classification'
)

training_args = TrainingArguments(
    output_dir="./logs/model_name",
    logging_dir="./logs/runs",
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=32,
    num_train_epochs=1,
    evaluate_during_training=True,
    logging_steps=25,
    eval_steps=250
)
set_seed(training_args.seed)

train_df, val_df, test_df = data.split_data()

model = MixedModelClassification(
    model_args = model_args,
    data_args = data_args,
    num_labels = len(column_info_dict['label_list']),
    train = train_df,
    validation = val_df,
    test = test_df
)

model.train(training_args = training_args)

