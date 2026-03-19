"""
Analisis Forense de Votos por Lider
====================================
Categoriza cada simpatizante segun la evidencia electoral en su mesa:

Categorias por mesa:
- CERO VOTOS:           0 votos para la candidata en esa mesa (incumplimiento total)
- VOTO SEGURO:          Solo este lider tiene simpatizantes en esa mesa Y hay votos >= simpatizantes
- INCUMPLIMIENTO PARCIAL: Solo este lider en esa mesa PERO votos < simpatizantes (algunos no votaron)
- VOTO PROBABLE:        Mesa compartida con otros lideres PERO votos >= total simpatizantes
- VOTO POSIBLE:         Mesa compartida, hay votos pero < total simpatizantes (no se puede atribuir)

Flags adicionales:
- EXCEDENTE:            Cuando votos > total simpatizantes (hay apoyo externo en esa mesa)
"""

import pandas as pd
import pickle, os, json
from pathlib import Path

# --- Reusar config y funciones de analisis_mesas ---
from analisis_mesas import (
    CANDIDATA_CODE, PARTIDO_CODE, TOTAL_VOTOS_CANDIDATA,
    MPIO_MAP, MPIO_NAME_TO_CODE, BASE_DIR,
    load_data, build_lugar_to_pto_map
)

# =========================================================
# 1. CONSTRUIR DETALLE POR MESA CON SIMPATIZANTES DE TODOS LOS LIDERES
# =========================================================

def construir_detalle_mesas(df_cam, df_sim_valle, lugar_pto_map):
    """
    Construye un dataframe con cada combinacion lider+puesto+mesa
    y la informacion de:
      - simpatizantes de ESE lider en esa mesa
      - simpatizantes de OTROS lideres en esa mesa
      - votos de la candidata en esa mesa
      - votos totales en esa mesa

    Estrategia de cruce:
    - Para Lugares con mapeo 'exact' a un pto: cruzar por mpio+zona+pto+mesa
    - Para todo lo demas: cruzar por mpio+zona+mesa sumando TODOS los ptos
      (ya que no sabemos en que pto esta el simpatizante, usamos la suma de
       votos de la candidata en todos los ptos de esa zona para esa mesa #)
    """
    # --- VOTOS POR MESA (electoral) ---
    cand_filter = (df_cam['candidato'] == CANDIDATA_CODE) & (df_cam['partido'] == PARTIDO_CODE)

    # Votos por pto+mesa (para mapeos exact)
    votos_cand_pto = (
        df_cam[cand_filter]
        .groupby(['mpio', 'zona', 'pto', 'mesa'])['votos']
        .sum().reset_index()
        .rename(columns={'votos': 'votos_candidata'})
    )
    votos_total_pto = (
        df_cam.groupby(['mpio', 'zona', 'pto', 'mesa'])['votos']
        .sum().reset_index()
        .rename(columns={'votos': 'votos_totales'})
    )
    votos_mesa_pto = votos_total_pto.merge(votos_cand_pto, on=['mpio', 'zona', 'pto', 'mesa'], how='left')
    votos_mesa_pto['votos_candidata'] = votos_mesa_pto['votos_candidata'].fillna(0).astype(int)

    # Votos por zona+mesa (sumando TODOS los ptos — para mapeos no-exact)
    votos_cand_zona = (
        df_cam[cand_filter]
        .groupby(['mpio', 'zona', 'mesa'])['votos']
        .sum().reset_index()
        .rename(columns={'votos': 'votos_candidata'})
    )
    votos_total_zona = (
        df_cam.groupby(['mpio', 'zona', 'mesa'])['votos']
        .sum().reset_index()
        .rename(columns={'votos': 'votos_totales'})
    )
    votos_mesa_zona = votos_total_zona.merge(votos_cand_zona, on=['mpio', 'zona', 'mesa'], how='left')
    votos_mesa_zona['votos_candidata'] = votos_mesa_zona['votos_candidata'].fillna(0).astype(int)

    # Numero de ptos por zona+mesa (para saber si la mesa esta en multiples ptos)
    n_ptos = (
        df_cam.groupby(['mpio', 'zona', 'mesa'])['pto'].nunique()
        .reset_index().rename(columns={'pto': 'n_ptos_zona'})
    )
    votos_mesa_zona = votos_mesa_zona.merge(n_ptos, on=['mpio', 'zona', 'mesa'], how='left')

    # --- SIMPATIZANTES POR MESA (por lider) ---
    sim_mesa = (
        df_sim_valle.dropna(subset=['mesa_num'])
        .groupby(['Líder', 'mpio_code', 'Municipio', 'zona_code', 'Lugar', 'mesa_num'])
        .agg(simp_lider=('Cédula', 'count'))
        .reset_index()
    )

    # --- TOTAL SIMPATIZANTES POR MESA (todos los lideres) ---
    total_simp_mesa = (
        df_sim_valle.dropna(subset=['mesa_num'])
        .groupby(['mpio_code', 'zona_code', 'Lugar', 'mesa_num'])
        .agg(simp_total=('Cédula', 'count'), n_lideres=('Líder', 'nunique'))
        .reset_index()
    )

    # Merge: cada lider-mesa + total simpatizantes en esa mesa
    detalle = sim_mesa.merge(
        total_simp_mesa,
        on=['mpio_code', 'zona_code', 'Lugar', 'mesa_num'],
        how='left'
    )
    detalle['simp_otros'] = detalle['simp_total'] - detalle['simp_lider']

    # --- CRUCE CON MAPEO Lugar -> pto ---
    detalle = detalle.merge(
        lugar_pto_map[['mpio', 'zona', 'lugar', 'pto', 'match_type']],
        left_on=['mpio_code', 'zona_code', 'Lugar'],
        right_on=['mpio', 'zona', 'lugar'],
        how='left'
    )

    # --- GRUPO 1: Mapeo 'exact' → cruzar por mpio+zona+pto+mesa ---
    exact = detalle[detalle['match_type'] == 'exact'].copy()
    exact['pto'] = exact['pto'].astype(int)
    exact['mesa_num'] = exact['mesa_num'].astype(int)

    result_exact = exact.merge(
        votos_mesa_pto,
        left_on=['mpio_code', 'zona_code', 'pto', 'mesa_num'],
        right_on=['mpio', 'zona', 'pto', 'mesa'],
        how='left',
        suffixes=('', '_elec')
    )
    result_exact['match_type'] = 'exact'

    # --- GRUPO 2: Todo lo demas → cruzar por mpio+zona+mesa (suma ptos) ---
    not_exact = detalle[detalle['match_type'] != 'exact'].copy()
    not_exact['mesa_num'] = not_exact['mesa_num'].astype(int)

    result_zona = not_exact.merge(
        votos_mesa_zona,
        left_on=['mpio_code', 'zona_code', 'mesa_num'],
        right_on=['mpio', 'zona', 'mesa'],
        how='left',
        suffixes=('', '_zona')
    )
    # Marcar tipo de cruce
    result_zona.loc[result_zona['match_type'].isna(), 'match_type'] = 'zona_mesa'
    result_zona.loc[result_zona['match_type'] == 'ambiguous', 'match_type'] = 'zona_mesa'
    result_zona.loc[result_zona['match_type'] == 'approx', 'match_type'] = 'zona_mesa'

    # --- COMBINAR ---
    cols = ['Líder', 'Municipio', 'zona_code', 'Lugar', 'mesa_num',
            'simp_lider', 'simp_otros', 'simp_total', 'n_lideres',
            'votos_candidata', 'votos_totales', 'match_type']

    for col in cols:
        if col not in result_exact.columns:
            result_exact[col] = 0
        if col not in result_zona.columns:
            result_zona[col] = 0

    result = pd.concat([
        result_exact[cols],
        result_zona[cols]
    ], ignore_index=True)

    result['votos_candidata'] = result['votos_candidata'].fillna(0).astype(int)
    result['votos_totales'] = result['votos_totales'].fillna(0).astype(int)
    result['simp_otros'] = result['simp_otros'].fillna(0).astype(int)
    result['simp_total'] = result['simp_total'].fillna(0).astype(int)
    result['n_lideres'] = result['n_lideres'].fillna(1).astype(int)

    return result.sort_values(['Líder', 'Municipio', 'zona_code', 'Lugar', 'mesa_num'])


# =========================================================
# 2. CLASIFICACION FORENSE
# =========================================================

def clasificar_forense(result):
    """
    Clasifica cada registro lider-mesa en una categoria forense.

    V  = votos_candidata en esa mesa
    Sl = simpatizantes de ESTE lider en esa mesa
    So = simpatizantes de OTROS lideres en esa mesa
    St = Sl + So (total simpatizantes en esa mesa)

    Categorias:
    1. CERO VOTOS:           V == 0
    2. VOTO SEGURO:          V > 0, So == 0, V >= Sl
    3. INCUMPLIMIENTO PARCIAL: V > 0, So == 0, V < Sl
    4. VOTO PROBABLE:        V > 0, So > 0, V >= St
    5. VOTO POSIBLE:         V > 0, So > 0, V < St

    Flag excedente: V > St (apoyo externo)
    """
    V = result['votos_candidata']
    Sl = result['simp_lider']
    So = result['simp_otros']
    St = result['simp_total']

    conditions = [
        V == 0,                             # CERO VOTOS
        (V > 0) & (So == 0) & (V >= Sl),   # VOTO SEGURO
        (V > 0) & (So == 0) & (V < Sl),    # INCUMPLIMIENTO PARCIAL
        (V > 0) & (So > 0) & (V >= St),    # VOTO PROBABLE
        (V > 0) & (So > 0) & (V < St),     # VOTO POSIBLE
    ]
    categorias = [
        'CERO VOTOS',
        'VOTO SEGURO',
        'INCUMPLIMIENTO PARCIAL',
        'VOTO PROBABLE',
        'VOTO POSIBLE',
    ]

    result['categoria'] = pd.Series(
        pd.Categorical(
            pd.array(['SIN CLASIFICAR'] * len(result)),
            categories=categorias + ['SIN CLASIFICAR']
        )
    )
    # Apply in order (numpy select-style)
    import numpy as np
    result['categoria'] = np.select(conditions, categorias, default='SIN CLASIFICAR')

    # Flag excedente
    result['excedente'] = (V > St).astype(int)
    result['votos_excedentes'] = (V - St).clip(lower=0)

    # Tasa de cumplimiento por mesa para este lider
    # votos atribuibles = min(V, Sl) cuando So==0, o proporcion cuando compartido
    result['votos_atribuidos'] = 0
    solo = So == 0
    result.loc[solo & (V > 0), 'votos_atribuidos'] = result.loc[solo & (V > 0), ['votos_candidata', 'simp_lider']].min(axis=1)

    compartido = So > 0
    # Proporcion: (Sl/St) * min(V, St) -> parte de votos que corresponde a este lider
    result.loc[compartido & (V > 0), 'votos_atribuidos'] = (
        (result.loc[compartido & (V > 0), 'simp_lider'] /
         result.loc[compartido & (V > 0), 'simp_total'].replace(0, 1)) *
        result.loc[compartido & (V > 0), ['votos_candidata', 'simp_total']].min(axis=1)
    ).round(0).astype(int)

    result['tasa_cumplimiento'] = (
        result['votos_atribuidos'] / result['simp_lider'].replace(0, float('nan'))
    ).round(4)

    return result


# =========================================================
# 3. RESUMENES FORENSES
# =========================================================

def resumen_forense_lider(result):
    """
    Resumen por lider con conteo de simpatizantes en cada categoria.
    """
    # Contar simpatizantes (no registros-mesa)
    cat_counts = (
        result.groupby(['Líder', 'categoria'])['simp_lider']
        .sum().unstack(fill_value=0).reset_index()
    )

    # Totales por lider
    totales = result.groupby('Líder').agg(
        total_simpatizantes=('simp_lider', 'sum'),
        mesas_presencia=('mesa_num', 'count'),
        mesas_cero=('votos_candidata', lambda x: (x == 0).sum()),
        votos_atribuidos=('votos_atribuidos', 'sum'),
        municipios=('Municipio', 'nunique'),
        puestos=('Lugar', 'nunique'),
    ).reset_index()

    resumen = totales.merge(cat_counts, on='Líder', how='left')

    # Porcentajes de cada categoria
    for cat in ['CERO VOTOS', 'VOTO SEGURO', 'INCUMPLIMIENTO PARCIAL', 'VOTO PROBABLE', 'VOTO POSIBLE']:
        if cat not in resumen.columns:
            resumen[cat] = 0
        col_pct = f'%_{cat.replace(" ", "_").lower()}'
        resumen[col_pct] = (resumen[cat] / resumen['total_simpatizantes'].replace(0, 1) * 100).round(1)

    # Cumplimiento global
    resumen['cumplimiento_%'] = (
        resumen['votos_atribuidos'] / resumen['total_simpatizantes'].replace(0, 1) * 100
    ).round(1)

    # Veredicto forense
    resumen['veredicto'] = 'REGULAR'
    resumen.loc[resumen['cumplimiento_%'] >= 70, 'veredicto'] = 'CUMPLIO'
    resumen.loc[resumen['cumplimiento_%'] < 40, 'veredicto'] = 'NO CUMPLIO'
    resumen.loc[
        (resumen['cumplimiento_%'] >= 40) & (resumen['cumplimiento_%'] < 70),
        'veredicto'
    ] = 'PARCIAL'

    return resumen.sort_values('total_simpatizantes', ascending=False)


def resumen_forense_detalle(result):
    """
    Detalle por lider + puesto con clasificacion forense.
    """
    detalle = result.groupby(['Líder', 'Municipio', 'Lugar', 'zona_code', 'categoria']).agg(
        simpatizantes=('simp_lider', 'sum'),
        votos_candidata=('votos_candidata', 'sum'),
        votos_atribuidos=('votos_atribuidos', 'sum'),
        mesas=('mesa_num', 'count'),
    ).reset_index()

    return detalle.sort_values(['Líder', 'Municipio', 'Lugar', 'categoria'])


def resumen_forense_mesa(result):
    """
    Detalle completo mesa por mesa (para auditar).
    """
    cols_out = [
        'Líder', 'Municipio', 'zona_code', 'Lugar', 'mesa_num',
        'simp_lider', 'simp_otros', 'simp_total', 'n_lideres',
        'votos_candidata', 'votos_totales',
        'categoria', 'votos_atribuidos', 'tasa_cumplimiento',
        'excedente', 'votos_excedentes', 'match_type'
    ]
    return result[cols_out].sort_values(
        ['Líder', 'Municipio', 'zona_code', 'Lugar', 'mesa_num']
    )


# =========================================================
# 4. EXPORTAR EXCEL
# =========================================================

def exportar_forense_excel(resumen_lider, detalle_puesto, detalle_mesa):
    """Export forensic analysis to Excel."""
    output_path = BASE_DIR / 'Analisis_Forense_Votos.xlsx'

    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        resumen_lider.to_excel(writer, sheet_name='Resumen Lideres', index=False)
        detalle_puesto.to_excel(writer, sheet_name='Detalle Puesto', index=False)
        detalle_mesa.to_excel(writer, sheet_name='Detalle Mesa', index=False)

        wb = writer.book
        # Formatos
        fmt_pct = wb.add_format({'num_format': '0.0%'})
        fmt_header = wb.add_format({
            'bold': True, 'bg_color': '#1a237e', 'font_color': 'white',
            'border': 1, 'text_wrap': True
        })
        fmt_verde = wb.add_format({'bg_color': '#c8e6c9'})
        fmt_rojo = wb.add_format({'bg_color': '#ffcdd2'})
        fmt_amarillo = wb.add_format({'bg_color': '#fff9c4'})
        fmt_naranja = wb.add_format({'bg_color': '#ffe0b2'})

        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            ws.set_column('A:Z', 16)

        # Formato condicional en Resumen Lideres para veredicto
        ws_lider = writer.sheets['Resumen Lideres']
        vered_col = list(resumen_lider.columns).index('veredicto')
        n_rows = len(resumen_lider) + 1
        ws_lider.conditional_format(1, vered_col, n_rows, vered_col, {
            'type': 'cell', 'criteria': '==', 'value': '"CUMPLIO"', 'format': fmt_verde
        })
        ws_lider.conditional_format(1, vered_col, n_rows, vered_col, {
            'type': 'cell', 'criteria': '==', 'value': '"NO CUMPLIO"', 'format': fmt_rojo
        })
        ws_lider.conditional_format(1, vered_col, n_rows, vered_col, {
            'type': 'cell', 'criteria': '==', 'value': '"PARCIAL"', 'format': fmt_naranja
        })

    print(f"  Exportado: {output_path}", flush=True)
    return output_path


# =========================================================
# 5. GENERAR JSON PARA DASHBOARD
# =========================================================

def generar_datos_forense(resumen_lider, detalle_mesa):
    """Generate JSON data for the forensic dashboard."""

    # --- Lideres con detalle de mesas ---
    lideres = []
    for _, r in resumen_lider.iterrows():
        lider_name = r['Líder']
        d = {
            'lider': lider_name,
            'total_simp': int(r['total_simpatizantes']),
            'votos_atribuidos': int(r['votos_atribuidos']),
            'cumplimiento': float(r['cumplimiento_%']),
            'veredicto': r['veredicto'],
            'mesas_presencia': int(r['mesas_presencia']),
            'mesas_cero': int(r['mesas_cero']),
            'municipios': int(r['municipios']),
            'puestos': int(r['puestos']),
            'categorias': {}
        }
        for cat in ['CERO VOTOS', 'VOTO SEGURO', 'INCUMPLIMIENTO PARCIAL', 'VOTO PROBABLE', 'VOTO POSIBLE']:
            d['categorias'][cat] = int(r.get(cat, 0))

        # --- Detalle mesa por mesa para este lider ---
        lider_rows = detalle_mesa[detalle_mesa['Líder'] == lider_name]
        mesas_detail = []
        for _, m in lider_rows.iterrows():
            mesas_detail.append({
                'mpio': str(m['Municipio']),
                'zona': int(m['zona_code']) if pd.notna(m['zona_code']) else 0,
                'lugar': str(m['Lugar']),
                'mesa': int(m['mesa_num']),
                'sl': int(m['simp_lider']),
                'so': int(m['simp_otros']),
                'st': int(m['simp_total']),
                'nl': int(m['n_lideres']),
                'vc': int(m['votos_candidata']),
                'vt': int(m['votos_totales']),
                'cat': str(m['categoria']),
                'va': int(m['votos_atribuidos']),
                'tc': float(m['tasa_cumplimiento']) if pd.notna(m['tasa_cumplimiento']) else 0,
                'exc': int(m['excedente']),
            })
        d['mesas'] = mesas_detail
        lideres.append(d)

    # --- Global ---
    total_simp = int(detalle_mesa['simp_lider'].sum())
    total_votos_attr = int(detalle_mesa['votos_atribuidos'].sum())
    total_mesas = len(detalle_mesa)
    mesas_cero = int((detalle_mesa['votos_candidata'] == 0).sum())
    mesas_excedente = int(detalle_mesa['excedente'].sum())

    cat_global = detalle_mesa.groupby('categoria')['simp_lider'].sum().to_dict()

    # Veredicto distribution
    veredictos = resumen_lider['veredicto'].value_counts().to_dict()

    global_stats = {
        'total_simpatizantes': total_simp,
        'total_votos_atribuidos': total_votos_attr,
        'cumplimiento_global': round(total_votos_attr / max(total_simp, 1) * 100, 1),
        'total_mesas': total_mesas,
        'mesas_cero_votos': mesas_cero,
        'mesas_excedente': mesas_excedente,
        'total_votos_candidata': TOTAL_VOTOS_CANDIDATA,
        'categorias': {k: int(v) for k, v in cat_global.items()},
        'veredictos': {k: int(v) for k, v in veredictos.items()},
    }

    return {'lideres': lideres, 'global': global_stats}


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 60, flush=True)
    print("  ANALISIS FORENSE DE VOTOS POR LIDER", flush=True)
    print("=" * 60, flush=True)

    print("\n1. Cargando datos...", flush=True)
    df_cam, df_sim_valle = load_data()

    print("\n2. Mapeando Lugar -> pto...", flush=True)
    lugar_pto_map = build_lugar_to_pto_map(df_cam, df_sim_valle)

    print("\n3. Construyendo detalle por mesa (todos los lideres)...", flush=True)
    detalle = construir_detalle_mesas(df_cam, df_sim_valle, lugar_pto_map)
    print(f"  {len(detalle):,} registros lider-mesa generados", flush=True)

    print("\n4. Clasificacion forense...", flush=True)
    detalle = clasificar_forense(detalle)

    # Mostrar distribucion
    cat_dist = detalle.groupby('categoria')['simp_lider'].sum()
    total_s = cat_dist.sum()
    print("  Distribucion de simpatizantes por categoria:", flush=True)
    for cat, n in cat_dist.items():
        print(f"    {cat}: {n:,} ({n/total_s*100:.1f}%)", flush=True)

    print("\n5. Generando resumenes...", flush=True)
    resumen_lider = resumen_forense_lider(detalle)
    detalle_puesto = resumen_forense_detalle(detalle)
    detalle_mesa = resumen_forense_mesa(detalle)

    print("\n6. Exportando Excel...", flush=True)
    exportar_forense_excel(resumen_lider, detalle_puesto, detalle_mesa)

    print("\n7. Generando datos para dashboard...", flush=True)
    dashboard_data = generar_datos_forense(resumen_lider, detalle)
    json_path = BASE_DIR / 'forense_data.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dashboard_data, f, ensure_ascii=False, indent=2)
    print(f"  JSON: {json_path}", flush=True)

    # --- RESUMEN IMPRESO ---
    print("\n" + "=" * 60, flush=True)
    print("  RESUMEN FORENSE", flush=True)
    print("=" * 60, flush=True)
    gs = dashboard_data['global']
    print(f"  Simpatizantes analizados:   {gs['total_simpatizantes']:,}", flush=True)
    print(f"  Votos atribuidos:           {gs['total_votos_atribuidos']:,}", flush=True)
    print(f"  Cumplimiento global:        {gs['cumplimiento_global']:.1f}%", flush=True)
    print(f"  Mesas con 0 votos:          {gs['mesas_cero_votos']:,} / {gs['total_mesas']:,}", flush=True)
    print(f"  Mesas con excedente:        {gs['mesas_excedente']:,}", flush=True)

    print("\n--- Veredicto por Lider ---", flush=True)
    for v in ['CUMPLIO', 'PARCIAL', 'NO CUMPLIO']:
        count = gs['veredictos'].get(v, 0)
        print(f"  {v}: {count} lideres", flush=True)

    print("\n--- Top 15 Lideres (por simpatizantes) ---", flush=True)
    for ld in dashboard_data['lideres'][:15]:
        cats = ld['categorias']
        print(
            f"  {ld['lider']}: {ld['total_simp']:,} simp | "
            f"cumpl={ld['cumplimiento']:.1f}% | {ld['veredicto']} | "
            f"Seguro={cats.get('VOTO SEGURO',0):,} "
            f"Prob={cats.get('VOTO PROBABLE',0):,} "
            f"Posib={cats.get('VOTO POSIBLE',0):,} "
            f"Parcial={cats.get('INCUMPLIMIENTO PARCIAL',0):,} "
            f"Cero={cats.get('CERO VOTOS',0):,}",
            flush=True
        )

    print("\nDONE", flush=True)
    return dashboard_data


if __name__ == '__main__':
    main()
