import os
from typing import List
from utils import mask_float

ENVIROMENT: str = "DEV"
ROOT_DIR: str = os.path.abspath(
    os.path.join(__file__, "../..")
) if ENVIROMENT == "DEV" else os.getcwd()

FINES_BASE_PROPERTIES = {
    "usecols": [8, 9],
    "skiprows": 1,
    "delimiter": ";",
    "names": ["contrato", "sancion"],
    "converters": {
        "contrato": str,
        "sancion": mask_float
    } 
}

GENERAL_BASE_PROPERTIES = {
    "usecols": [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 17, 18, 19, 20, 28, 30, 31, 36, 38, 41, 43, 45, 46, 51, 52],    
    "skiprows": 1,
    "names": ['nivel_entidad',
        'orden_entidad',
        'tipo_de_proceso',
        'estado_del_proceso',
        'otras_formas_contratacion',
        'regimen_de_contratacion',
        'objeto_a_contratar',
        'tipo_de_contrato',
        'municipio_obtencion',
        'municipio_entrega',
        'municipios_ejecucion',
        'detalle_del_objeto_a_contratar',
        'contrato',
        'cuantia_proceso',
        'nombre_grupo',
        'nombre_familia',
        'nombre_clase',
        'plazo_de_ejec_del_contrato',
        'tiempo_adiciones_en_dias',
        'tiempo_adiciones_en_meses',
        'valor_contrato_con_adiciones',
        'origen_de_los_recursos',
        'calificacion_definitiva',
        'moneda',
        'marcacion_adiciones',
        'nombre_rubro',
        'municipio_entidad',
        'departamento_entidad'],
    "converters": {
        "contrato": str
    } 
}

NO_REQUIRED_COLUMNS: List[str] = [
    'anno_cargue_secop',
    'nombre_de_la_entidad',
    'numero_de_constancia',
    'numero_de_proceso',
    'id_ajudicacion',
    'tipo_identifi_del_contratista',
    'nom_raz_social_contratista',
    'tipo_doc_representante_legal',
    'nombre_del_represen_legal',
    'dpto_y_muni_contratista',
    'fecha_de_firma_del_contrato',
    'fecha_ini_ejec_contrato',
    'fecha_fin_ejec_contrato',
    'compromiso_presupuestal',
    'cuantia_contrato',
    'valor_total_de_adiciones', #evaluate
    'objeto_del_contrato_a_la_firma',
    'codigo_bpin',
    'espostconflicto',
    'proponentes_seleccionados',
    'nombre_sub_unidad_ejecutora',
    'valor_rubro',
    'sexo_replegal_entidad',
    'pilar_acuerdo_paz',
    'punto_acuerdo_paz',
]

