"""
Analisis de Votaciones - Camara de Representantes, Valle del Cauca
=================================================================

Cruza resultados electorales (CAMARA valle.xlsx) con base de simpatizantes
(Simpatizantes.xlsx) para medir la efectividad de la candidata por zona,
municipio y lider.

Uso:
    python analisis_votaciones.py

Genera: Resultado_Analisis_Votaciones.xlsx con multiples hojas de analisis.
"""

import pandas as pd
import sys
from pathlib import Path

# --- CONFIGURACION ---
CANDIDATA_CODE = 107
CANDIDATA_NOMBRE = 'Ana Maria Sanclemente'
PARTIDO_CODE = 3050
PARTIDO_NOMBRE = 'CAMBIO RADICAL/ALMA'

MPIO_MAP = {
    1: 'CALI', 4: 'ALCALA', 7: 'ANDALUCIA', 10: 'ANSERMANUEVO',
    13: 'ARGELIA', 16: 'BOLIVAR', 19: 'BUENAVENTURA', 22: 'BUGA',
    25: 'BUGALAGRANDE', 28: 'CAICEDONIA', 31: 'CALIMA', 34: 'CANDELARIA',
    37: 'CARTAGO', 40: 'DAGUA', 43: 'EL AGUILA', 46: 'EL CAIRO',
    49: 'EL CERRITO', 52: 'EL DOVIO', 55: 'FLORIDA', 58: 'GINEBRA',
    61: 'GUACARI', 64: 'JAMUNDI', 67: 'LA CUMBRE', 70: 'LA UNION',
    73: 'LA VICTORIA', 76: 'OBANDO', 79: 'PALMIRA', 82: 'PRADERA',
    85: 'RESTREPO', 88: 'RIOFRIO', 91: 'ROLDANILLO', 94: 'SAN PEDRO',
    97: 'SEVILLA', 100: 'TORO', 103: 'TRUJILLO', 106: 'TULUA',
    109: 'ULLOA', 112: 'VERSALLES', 115: 'VIJES', 118: 'YOTOCO',
    121: 'YUMBO', 124: 'ZARZAL'
}
MPIO_NAME_TO_CODE = {v: k for k, v in MPIO_MAP.items()}

PARTIDO_MAP = {
    1: 'LIBERAL', 2: 'CONSERVADOR', 4: 'ALIANZA VERDE', 5: 'AICO',
    8: 'LA U', 11: 'CENTRO DEMOCRATICO',
    20: 'SALVACION NACIONAL', 21: 'PARTIDO OXIGENO',
    24: 'DEMOCRATA COLOMBIANO',
    3050: 'CAMBIO R-ALMA', 3051: 'MIRA-NVO LIB',
    3057: 'PACTO HISTORICO', 3113: 'FRENTE AMPLIO', 3137: 'FUERZA CIUDADANA'
}

CODIGOS_ESPECIALES = {996: 'VOTO EN BLANCO', 997: 'VOTOS NULOS', 998: 'TARJETAS NO MARCADAS', 0: 'VOTO POR LISTA'}


def cargar_datos(camara_path, simpatizantes_path):
    """Carga y limpia ambos archivos."""
    df_camara = pd.read_excel(camara_path, sheet_name=0)
    df_sim = pd.read_excel(simpatizantes_path)

    # Enriquecer CAMARA
    df_camara['municipio_nombre'] = df_camara['mpio'].map(MPIO_MAP)
    df_camara['partido_nombre'] = df_camara['partido'].map(PARTIDO_MAP).fillna('OTRO')
    df_camara['tipo_voto'] = df_camara['candidato'].map(CODIGOS_ESPECIALES).fillna('PREFERENTE')

    # Limpiar Simpatizantes
    for col in ['Municipio', 'Lugar', 'Departamento']:
        if col in df_sim.columns:
            df_sim[col] = df_sim[col].astype(str).str.strip().str.upper()
            df_sim.loc[df_sim[col] == 'NAN', col] = None

    df_sim['mpio_code'] = df_sim['Municipio'].map(MPIO_NAME_TO_CODE)
    df_sim['zona_code'] = df_sim['Comuna']

    df_sim_valle = df_sim[df_sim['Departamento'] == 'VALLE'].copy()

    return df_camara, df_sim, df_sim_valle


def analisis_por_zona(df_camara, df_sim_valle):
    """Tabla de analisis por zona (municipio + zona/comuna)."""
    votos_zona_total = (df_camara
        .groupby(['mpio', 'municipio_nombre', 'zona'])['votos']
        .sum().reset_index()
        .rename(columns={'votos': 'votos_totales_zona'}))

    votos_candidata_zona = (df_camara[df_camara['candidato'] == CANDIDATA_CODE]
        .groupby(['mpio', 'municipio_nombre', 'zona'])['votos']
        .sum().reset_index()
        .rename(columns={'votos': 'votos_candidata_zona'}))

    votos_partido_zona = (df_camara[df_camara['partido'] == PARTIDO_CODE]
        .groupby(['mpio', 'municipio_nombre', 'zona'])['votos']
        .sum().reset_index()
        .rename(columns={'votos': 'votos_partido_zona'}))

    sim_zona = (df_sim_valle
        .groupby(['Municipio', 'zona_code'])
        .agg(simpatizantes=('Cédula', 'count'), puestos_unicos=('Lugar', 'nunique'))
        .reset_index())

    votos_merged = votos_zona_total.merge(
        votos_candidata_zona, on=['mpio', 'municipio_nombre', 'zona'], how='left'
    ).merge(
        votos_partido_zona, on=['mpio', 'municipio_nombre', 'zona'], how='left')

    votos_merged['votos_candidata_zona'] = votos_merged['votos_candidata_zona'].fillna(0).astype(int)
    votos_merged['votos_partido_zona'] = votos_merged['votos_partido_zona'].fillna(0).astype(int)

    analisis = votos_merged.merge(
        sim_zona, left_on=['municipio_nombre', 'zona'],
        right_on=['Municipio', 'zona_code'], how='outer', indicator=True)

    analisis['simpatizantes'] = analisis['simpatizantes'].fillna(0).astype(int)
    analisis['votos_totales_zona'] = analisis['votos_totales_zona'].fillna(0).astype(int)
    analisis['votos_candidata_zona'] = analisis['votos_candidata_zona'].fillna(0).astype(int)
    analisis['municipio_nombre'] = analisis['municipio_nombre'].fillna(analisis['Municipio'])

    analisis['ratio_candidata_vs_simpatizantes'] = (
        analisis['votos_candidata_zona'] / analisis['simpatizantes'].replace(0, float('nan'))).round(2)
    analisis['pct_candidata_en_zona'] = (
        analisis['votos_candidata_zona'] / analisis['votos_totales_zona'].replace(0, float('nan')) * 100).round(2)

    return analisis


def analisis_por_municipio(df_camara, df_sim_valle):
    """Tabla resumen por municipio."""
    votos_total = df_camara.groupby(['mpio', 'municipio_nombre'])['votos'].sum().reset_index()
    votos_total.columns = ['mpio', 'municipio', 'votos_totales']

    votos_cand = (df_camara[df_camara['candidato'] == CANDIDATA_CODE]
        .groupby('municipio_nombre')['votos'].sum().reset_index())
    votos_cand.columns = ['municipio', 'votos_candidata']

    votos_part = (df_camara[df_camara['partido'] == PARTIDO_CODE]
        .groupby('municipio_nombre')['votos'].sum().reset_index())
    votos_part.columns = ['municipio', 'votos_partido']

    sim_mpio = (df_sim_valle.groupby('Municipio')
        .agg(simpatizantes=('Cédula', 'count'), puestos_unicos=('Lugar', 'nunique'),
             lideres_unicos=('Líder', 'nunique'))
        .reset_index().rename(columns={'Municipio': 'municipio'}))

    resumen = votos_total.merge(votos_cand, on='municipio', how='left')
    resumen = resumen.merge(votos_part, on='municipio', how='left')
    resumen = resumen.merge(sim_mpio, on='municipio', how='outer')
    resumen['votos_candidata'] = resumen['votos_candidata'].fillna(0).astype(int)
    resumen['votos_partido'] = resumen['votos_partido'].fillna(0).astype(int)
    resumen['simpatizantes'] = resumen['simpatizantes'].fillna(0).astype(int)
    resumen['ratio_votos_vs_simp'] = (
        resumen['votos_candidata'] / resumen['simpatizantes'].replace(0, float('nan'))).round(2)
    resumen['pct_candidata'] = (
        resumen['votos_candidata'] / resumen['votos_totales'].replace(0, float('nan')) * 100).round(2)

    return resumen.sort_values('simpatizantes', ascending=False)


def analisis_por_puesto(df_sim_valle, analisis_zona):
    """Simpatizantes agrupados por puesto de votacion."""
    sim_puesto = (df_sim_valle
        .groupby(['Municipio', 'Lugar', 'zona_code'])
        .agg(simpatizantes=('Cédula', 'count'), lideres=('Líder', 'nunique'),
             lista_lideres=('Líder', lambda x: ', '.join(sorted(x.unique()))))
        .reset_index()
        .sort_values('simpatizantes', ascending=False))

    sim_puesto = sim_puesto.merge(
        analisis_zona[['municipio_nombre', 'zona', 'votos_totales_zona', 'votos_candidata_zona', 'votos_partido_zona']],
        left_on=['Municipio', 'zona_code'], right_on=['municipio_nombre', 'zona'], how='left')

    return sim_puesto


def analisis_por_lider(df_sim_valle, analisis_zona):
    """Efectividad por lider."""
    lider_sim = (df_sim_valle.groupby('Líder')
        .agg(simpatizantes=('Cédula', 'count'), municipios=('Municipio', 'nunique'),
             puestos=('Lugar', 'nunique'),
             lista_municipios=('Municipio', lambda x: ', '.join(sorted(x.unique()))))
        .reset_index())

    lider_zonas = (df_sim_valle.groupby(['Líder', 'Municipio', 'zona_code']).size()
        .reset_index(name='n_simp'))
    lider_zonas = lider_zonas.merge(
        analisis_zona[['municipio_nombre', 'zona', 'votos_candidata_zona', 'votos_partido_zona']],
        left_on=['Municipio', 'zona_code'], right_on=['municipio_nombre', 'zona'], how='left')

    lider_votos = lider_zonas.groupby('Líder').agg(
        votos_candidata_en_zonas=('votos_candidata_zona', 'sum'),
        votos_partido_en_zonas=('votos_partido_zona', 'sum')).reset_index()

    lider = lider_sim.merge(lider_votos, on='Líder', how='left')
    lider['ratio_votos_candidata_vs_simp'] = (
        lider['votos_candidata_en_zonas'] / lider['simpatizantes'].replace(0, float('nan'))).round(2)

    return lider.sort_values('simpatizantes', ascending=False)


def ranking_candidatos(df_camara, df_sim_valle):
    """Ranking de candidatos en zonas con simpatizantes."""
    zonas = df_sim_valle[['Municipio', 'zona_code']].drop_duplicates()
    zonas = zonas.rename(columns={'Municipio': 'municipio_nombre', 'zona_code': 'zona'})

    df_en_zonas = df_camara.merge(zonas, on=['municipio_nombre', 'zona'], how='inner')

    ranking = (df_en_zonas[df_en_zonas['tipo_voto'] == 'PREFERENTE']
        .groupby(['candidato', 'partido', 'partido_nombre'])['votos']
        .sum().reset_index()
        .sort_values('votos', ascending=False))

    ranking['es_candidata'] = ranking['candidato'] == CANDIDATA_CODE
    return ranking


def exportar(output_path, resumen_mpio, analisis_zona, puestos, lideres, ranking, df_sim_valle):
    """Exporta todo a Excel."""
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        resumen_mpio.to_excel(writer, sheet_name='Resumen Municipio', index=False)

        zona_exp = analisis_zona[analisis_zona['simpatizantes'] > 0]
        cols_zona = ['municipio_nombre', 'zona', 'simpatizantes', 'votos_totales_zona',
                     'votos_candidata_zona', 'votos_partido_zona',
                     'ratio_candidata_vs_simpatizantes', 'pct_candidata_en_zona']
        zona_exp = zona_exp[[c for c in cols_zona if c in zona_exp.columns]].sort_values(['municipio_nombre', 'zona'])
        zona_exp.to_excel(writer, sheet_name='Detalle por Zona', index=False)

        cols_p = ['Municipio', 'Lugar', 'zona_code', 'simpatizantes', 'lideres',
                  'votos_totales_zona', 'votos_candidata_zona', 'lista_lideres']
        puestos[[c for c in cols_p if c in puestos.columns]].to_excel(
            writer, sheet_name='Puestos Votacion', index=False)

        cols_l = ['Líder', 'simpatizantes', 'municipios', 'puestos',
                  'votos_candidata_en_zonas', 'votos_partido_en_zonas',
                  'ratio_votos_candidata_vs_simp', 'lista_municipios']
        lideres[[c for c in cols_l if c in lideres.columns]].to_excel(
            writer, sheet_name='Efectividad Lideres', index=False)

        ranking.to_excel(writer, sheet_name='Ranking Candidatos', index=False)

        sim_exp = df_sim_valle[['Cédula', 'Líder', 'Municipio', 'Lugar', 'zona_code', 'Barrio', 'Mesa']].copy()
        sim_exp.rename(columns={'zona_code': 'Comuna'}).to_excel(
            writer, sheet_name='Simpatizantes Detalle', index=False)

        for sheet_name in writer.sheets:
            writer.sheets[sheet_name].set_column('A:Z', 18)

    print(f'Exportado: {output_path}')


def main():
    base_dir = Path(__file__).parent
    camara_path = base_dir / 'CAMARA valle.xlsx'
    simpatizantes_path = base_dir / 'Simpatizantes.xlsx'
    output_path = base_dir / 'Resultado_Analisis_Votaciones.xlsx'

    print(f'Cargando datos...')
    df_camara, df_sim, df_sim_valle = cargar_datos(camara_path, simpatizantes_path)
    print(f'  CAMARA: {len(df_camara):,} registros')
    print(f'  Simpatizantes Valle: {len(df_sim_valle):,} registros')

    print(f'Analizando por zona...')
    az = analisis_por_zona(df_camara, df_sim_valle)

    print(f'Analizando por municipio...')
    rm = analisis_por_municipio(df_camara, df_sim_valle)

    print(f'Analizando por puesto...')
    puestos = analisis_por_puesto(df_sim_valle, az)

    print(f'Analizando por lider...')
    lideres = analisis_por_lider(df_sim_valle, az)

    print(f'Ranking candidatos...')
    rank = ranking_candidatos(df_camara, df_sim_valle)

    print(f'Exportando resultados...')
    exportar(output_path, rm, az, puestos, lideres, rank, df_sim_valle)

    # Resumen
    total_simp = df_sim_valle.shape[0]
    total_votos_cand = df_camara[df_camara['candidato'] == CANDIDATA_CODE]['votos'].sum()
    print(f'\n=== RESUMEN FINAL ===')
    print(f'Total simpatizantes Valle: {total_simp:,}')
    print(f'Total votos {CANDIDATA_NOMBRE}: {total_votos_cand:,}')
    print(f'Ratio global: {total_votos_cand / max(total_simp, 1):.2f}')


if __name__ == '__main__':
    main()
