"""
Analisis de Probabilidad de Voto por Mesa
==========================================
Cruza simpatizantes por lider+puesto+mesa con resultados electorales
a nivel de mpio+zona+pto+mesa para calcular probabilidad de voto real.

Estrategia de cruce:
- Los datos electorales usan codigo numerico 'pto' para puestos
- Los simpatizantes usan nombre texto 'Lugar' para puestos
- Intentamos mapear Lugar -> pto usando la cantidad de mesas como fingerprint
- Donde el mapeo es ambiguo, usamos el promedio de votos por mesa en la zona
"""

import pandas as pd
import pickle, os, json
from pathlib import Path

# --- CONFIG ---
CANDIDATA_CODE = 107
PARTIDO_CODE = 3050
TOTAL_VOTOS_CANDIDATA = 30237

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

BASE_DIR = Path(__file__).parent
CACHE_CAM = '/tmp/cam_cache.pkl'
CACHE_SIM = '/tmp/sim_cache.pkl'


def load_data():
    """Load data from cache or xlsx."""
    if os.path.exists(CACHE_CAM):
        df_cam = pickle.load(open(CACHE_CAM, 'rb'))
    else:
        print("Loading CAMARA from xlsx (slow, caching)...", flush=True)
        df_cam = pd.read_excel(BASE_DIR / 'CAMARA valle.xlsx', sheet_name=0)
        pickle.dump(df_cam, open(CACHE_CAM, 'wb'))

    if os.path.exists(CACHE_SIM):
        df_sim = pickle.load(open(CACHE_SIM, 'rb'))
    else:
        print("Loading Simpatizantes from xlsx (slow, caching)...", flush=True)
        df_sim = pd.read_excel(BASE_DIR / 'Registros_simpatizantes_13Mar.xlsx')
        pickle.dump(df_sim, open(CACHE_SIM, 'wb'))

    # Enrich electoral
    df_cam['municipio_nombre'] = df_cam['mpio'].map(MPIO_MAP)

    # Clean sympathizers
    for col in ['Municipio', 'Lugar', 'Departamento']:
        if col in df_sim.columns:
            df_sim[col] = df_sim[col].astype(str).str.strip().str.upper()
            df_sim.loc[df_sim[col] == 'NAN', col] = None

    mpio_fixes = {'CALIMA (DARIEN)': 'CALIMA', 'CALIMA(DARIEN)': 'CALIMA'}
    df_sim['Municipio'] = df_sim['Municipio'].replace(mpio_fixes)
    df_sim['mpio_code'] = df_sim['Municipio'].map(MPIO_NAME_TO_CODE)
    df_sim['zona_code'] = pd.to_numeric(df_sim['Comuna'], errors='coerce')
    df_sim['mesa_num'] = pd.to_numeric(df_sim['Mesa'], errors='coerce')

    df_sim_valle = df_sim[df_sim['Departamento'] == 'VALLE'].copy()

    # --- Inferir zona_code desde Lugar cuando Comuna está vacía ---
    has_zona = (df_sim_valle[df_sim_valle['zona_code'].notna()]
                .groupby(['Municipio', 'Lugar'])['zona_code']
                .agg(nunique='nunique', zona='first')
                .reset_index())
    lugar_zona_map = has_zona[has_zona['nunique'] == 1][['Municipio', 'Lugar', 'zona']].copy()

    df_sim_valle = df_sim_valle.merge(lugar_zona_map, on=['Municipio', 'Lugar'], how='left')
    df_sim_valle['zona_code'] = df_sim_valle['zona_code'].fillna(df_sim_valle['zona'])
    df_sim_valle.drop(columns=['zona'], inplace=True)

    n_con_zona = df_sim_valle['zona_code'].notna().sum()
    print(f"  CAMARA: {len(df_cam):,} rows | Simpatizantes Valle: {len(df_sim_valle):,} rows", flush=True)
    print(f"  Simpatizantes con zona (despues de inferir): {n_con_zona:,} / {len(df_sim_valle):,} ({n_con_zona/len(df_sim_valle)*100:.1f}%)", flush=True)
    return df_cam, df_sim_valle


def build_lugar_to_pto_map(df_cam, df_sim_valle):
    """
    Try to map Lugar (name) -> pto (code) within each mpio+zona.
    
    Strategy: count max mesa per pto in electoral and max mesa per Lugar in sympathizers.
    If within a zone, a Lugar's max mesa count uniquely matches one pto's max mesa count,
    we assign that mapping. Otherwise mark as ambiguous.
    """
    # Electoral: max mesa per mpio+zona+pto
    elec_mesas = (df_cam.groupby(['mpio', 'zona', 'pto'])['mesa']
                  .max().reset_index()
                  .rename(columns={'mesa': 'max_mesa_elec'}))

    # Sympathizers: max mesa per mpio+zona+Lugar
    sim_mesas = (df_sim_valle.dropna(subset=['mesa_num'])
                 .groupby(['mpio_code', 'zona_code', 'Lugar'])['mesa_num']
                 .agg(max_mesa_sim='max', count_sim='count')
                 .reset_index())

    mappings = []
    ambiguous_count = 0
    matched_count = 0

    for (mpio, zona), grp_elec in elec_mesas.groupby(['mpio', 'zona']):
        grp_sim = sim_mesas[(sim_mesas['mpio_code'] == mpio) & (sim_mesas['zona_code'] == zona)]
        if grp_sim.empty:
            continue

        # Build a dict: max_mesa -> list of ptos
        mesa_to_ptos = {}
        for _, row in grp_elec.iterrows():
            mesa_to_ptos.setdefault(int(row['max_mesa_elec']), []).append(int(row['pto']))

        for _, sim_row in grp_sim.iterrows():
            lugar = sim_row['Lugar']
            max_mesa_sim = int(sim_row['max_mesa_sim'])

            # Try exact match
            candidates = mesa_to_ptos.get(max_mesa_sim, [])
            if len(candidates) == 1:
                mappings.append({
                    'mpio': mpio, 'zona': zona, 'lugar': lugar,
                    'pto': candidates[0], 'match_type': 'exact'
                })
                matched_count += 1
            else:
                # Try closest match (sim may not have all mesas)
                # Find pto with closest max_mesa >= max_mesa_sim
                best_pto = None
                best_diff = 999
                for max_m, ptos in mesa_to_ptos.items():
                    if max_m >= max_mesa_sim and len(ptos) == 1:
                        diff = max_m - max_mesa_sim
                        if diff < best_diff:
                            best_diff = diff
                            best_pto = ptos[0]
                if best_pto is not None and best_diff <= 3:
                    mappings.append({
                        'mpio': mpio, 'zona': zona, 'lugar': lugar,
                        'pto': best_pto, 'match_type': 'approx'
                    })
                    matched_count += 1
                else:
                    mappings.append({
                        'mpio': mpio, 'zona': zona, 'lugar': lugar,
                        'pto': None, 'match_type': 'ambiguous'
                    })
                    ambiguous_count += 1

    map_df = pd.DataFrame(mappings)
    print(f"  Lugar->pto mapping: {matched_count} matched, {ambiguous_count} ambiguous", flush=True)
    return map_df


def analisis_por_mesa(df_cam, df_sim_valle, lugar_pto_map):
    """
    Build mesa-level analysis:
    For each sympathizer record (lider + municipio + lugar + mesa),
    find the actual votes for the candidata at that mesa.

    Estrategia de cruce:
    - Mapeos 'exact': cruzar por mpio+zona+pto+mesa
    - Todo lo demas: cruzar por mpio+zona+mesa sumando TODOS los ptos
    """
    cand_filter = (df_cam['candidato'] == CANDIDATA_CODE) & (df_cam['partido'] == PARTIDO_CODE)

    # --- VOTOS POR PTO+MESA (para mapeos exact) ---
    votos_cand_pto = (
        df_cam[cand_filter]
        .groupby(['mpio', 'zona', 'pto', 'mesa'])['votos']
        .sum().reset_index()
        .rename(columns={'votos': 'votos_candidata_mesa'})
    )
    votos_total_pto = (
        df_cam.groupby(['mpio', 'zona', 'pto', 'mesa'])['votos']
        .sum().reset_index()
        .rename(columns={'votos': 'votos_totales_mesa'})
    )
    votos_mesa_pto = votos_total_pto.merge(votos_cand_pto, on=['mpio', 'zona', 'pto', 'mesa'], how='left')
    votos_mesa_pto['votos_candidata_mesa'] = votos_mesa_pto['votos_candidata_mesa'].fillna(0).astype(int)

    # --- VOTOS POR ZONA+MESA (sumando todos los ptos — para no-exact) ---
    votos_cand_zona = (
        df_cam[cand_filter]
        .groupby(['mpio', 'zona', 'mesa'])['votos']
        .sum().reset_index()
        .rename(columns={'votos': 'votos_candidata_mesa'})
    )
    votos_total_zona = (
        df_cam.groupby(['mpio', 'zona', 'mesa'])['votos']
        .sum().reset_index()
        .rename(columns={'votos': 'votos_totales_mesa'})
    )
    votos_mesa_zona = votos_total_zona.merge(votos_cand_zona, on=['mpio', 'zona', 'mesa'], how='left')
    votos_mesa_zona['votos_candidata_mesa'] = votos_mesa_zona['votos_candidata_mesa'].fillna(0).astype(int)

    # --- SIMPATIZANTES POR MESA ---
    sim_mesa = (df_sim_valle.dropna(subset=['mesa_num'])
                .groupby(['Líder', 'mpio_code', 'Municipio', 'zona_code', 'Lugar', 'mesa_num'])
                .agg(simpatizantes=('Cédula', 'count'))
                .reset_index())

    # --- CRUCE: agregar pto mapping ---
    sim_mesa = sim_mesa.merge(
        lugar_pto_map[['mpio', 'zona', 'lugar', 'pto', 'match_type']],
        left_on=['mpio_code', 'zona_code', 'Lugar'],
        right_on=['mpio', 'zona', 'lugar'],
        how='left'
    )

    # --- GRUPO 1: Mapeo 'exact' → cruzar por mpio+zona+pto+mesa ---
    exact = sim_mesa[sim_mesa['match_type'] == 'exact'].copy()
    exact['pto'] = exact['pto'].astype(int)
    exact['mesa_num'] = exact['mesa_num'].astype(int)

    result_exact = exact.merge(
        votos_mesa_pto,
        left_on=['mpio_code', 'zona_code', 'pto', 'mesa_num'],
        right_on=['mpio', 'zona', 'pto', 'mesa'],
        how='left'
    )
    result_exact['match_type'] = 'exact'

    # --- GRUPO 2: Todo lo demas → cruzar por mpio+zona+mesa (suma ptos) ---
    not_exact = sim_mesa[sim_mesa['match_type'] != 'exact'].copy()
    not_exact['mesa_num'] = not_exact['mesa_num'].astype(int)

    result_zona = not_exact.merge(
        votos_mesa_zona,
        left_on=['mpio_code', 'zona_code', 'mesa_num'],
        right_on=['mpio', 'zona', 'mesa'],
        how='left',
        suffixes=('', '_zona')
    )
    result_zona.loc[result_zona['match_type'].isna(), 'match_type'] = 'zona_mesa'
    result_zona.loc[result_zona['match_type'] == 'ambiguous', 'match_type'] = 'zona_mesa'
    result_zona.loc[result_zona['match_type'] == 'approx', 'match_type'] = 'zona_mesa'

    # --- Combinar ---
    cols_final = ['Líder', 'Municipio', 'zona_code', 'Lugar', 'mesa_num',
                  'simpatizantes', 'votos_candidata_mesa', 'votos_totales_mesa', 'match_type']

    for col in cols_final:
        if col not in result_exact.columns:
            result_exact[col] = 0
        if col not in result_zona.columns:
            result_zona[col] = 0

    result = pd.concat([
        result_exact[cols_final],
        result_zona[cols_final]
    ], ignore_index=True)

    result['votos_candidata_mesa'] = result['votos_candidata_mesa'].fillna(0).astype(int)
    result['votos_totales_mesa'] = result['votos_totales_mesa'].fillna(0).astype(int)

    # --- Calcular metricas ---
    result['tiene_votos'] = (result['votos_candidata_mesa'] > 0).astype(int)
    result['votos_atribuidos'] = result[['votos_candidata_mesa', 'simpatizantes']].min(axis=1)
    result['prob_voto'] = (result['votos_atribuidos'] / result['simpatizantes'].replace(0, float('nan'))).round(4)
    result['excedente'] = (result['votos_candidata_mesa'] - result['simpatizantes']).clip(lower=0)

    return result.sort_values(['Líder', 'Municipio', 'zona_code', 'Lugar', 'mesa_num'])


def resumen_por_lider(result):
    """Resumen de probabilidad por lider."""
    # Create a unique mesa ID per location
    result['mesa_id'] = result['Municipio'] + '_' + result['zona_code'].astype(str) + '_' + result['Lugar'] + '_' + result['mesa_num'].astype(str)

    lider_stats = result.groupby('Líder').agg(
        simpatizantes=('simpatizantes', 'sum'),
        mesas_con_presencia=('mesa_id', 'nunique'),
        mesas_con_votos=('tiene_votos', 'sum'),
        votos_candidata_total=('votos_candidata_mesa', 'sum'),
        votos_atribuidos=('votos_atribuidos', 'sum'),
        municipios=('Municipio', 'nunique'),
        puestos=('Lugar', 'nunique'),
    ).reset_index()

    lider_stats['mesas_con_votos'] = lider_stats['mesas_con_votos'].astype(int)
    lider_stats['pct_mesas_con_votos'] = (
        lider_stats['mesas_con_votos'] / lider_stats['mesas_con_presencia'] * 100
    ).round(1)
    # Probabilidad = votos atribuidos / simpatizantes (capped at 1.0 per mesa, then aggregated)
    lider_stats['prob_voto'] = (
        lider_stats['votos_atribuidos'] / lider_stats['simpatizantes'].replace(0, float('nan'))
    ).round(4)
    # Ratio bruto (puede ser > 1.0)
    lider_stats['ratio_votos_simp'] = (
        lider_stats['votos_candidata_total'] / lider_stats['simpatizantes'].replace(0, float('nan'))
    ).round(2)

    return lider_stats.sort_values('simpatizantes', ascending=False)


def resumen_por_puesto(result):
    """Resumen por puesto de votacion."""
    result['mesa_id'] = result['Municipio'] + '_' + result['zona_code'].astype(str) + '_' + result['Lugar'] + '_' + result['mesa_num'].astype(str)
    puesto_stats = result.groupby(['Municipio', 'Lugar', 'zona_code']).agg(
        simpatizantes=('simpatizantes', 'sum'),
        mesas_presencia=('mesa_id', 'nunique'),
        mesas_con_votos=('tiene_votos', 'sum'),
        votos_candidata_sum=('votos_candidata_mesa', 'sum'),
        lideres=('Líder', 'nunique'),
    ).reset_index()

    puesto_stats['mesas_con_votos'] = puesto_stats['mesas_con_votos'].astype(int)
    puesto_stats['pct_mesas_con_votos'] = (
        puesto_stats['mesas_con_votos'] / puesto_stats['mesas_presencia'] * 100
    ).round(1)
    puesto_stats['votos_por_simp'] = (
        puesto_stats['votos_candidata_sum'] / puesto_stats['simpatizantes'].replace(0, float('nan'))
    ).round(2)

    return puesto_stats.sort_values('simpatizantes', ascending=False)


def resumen_por_lider_puesto(result):
    """Resumen por lider+puesto."""
    result['mesa_id'] = result['Municipio'] + '_' + result['zona_code'].astype(str) + '_' + result['Lugar'] + '_' + result['mesa_num'].astype(str)
    lp_stats = result.groupby(['Líder', 'Municipio', 'Lugar', 'zona_code']).agg(
        simpatizantes=('simpatizantes', 'sum'),
        mesas_presencia=('mesa_id', 'nunique'),
        mesas_con_votos=('tiene_votos', 'sum'),
        votos_candidata_sum=('votos_candidata_mesa', 'sum'),
    ).reset_index()

    lp_stats['mesas_con_votos'] = lp_stats['mesas_con_votos'].astype(int)
    lp_stats['pct_mesas_con_votos'] = (
        lp_stats['mesas_con_votos'] / lp_stats['mesas_presencia'] * 100
    ).round(1)
    lp_stats['votos_por_simp'] = (
        lp_stats['votos_candidata_sum'] / lp_stats['simpatizantes'].replace(0, float('nan'))
    ).round(2)

    return lp_stats.sort_values(['Líder', 'simpatizantes'], ascending=[True, False])


def exportar_excel(result, lider_stats, puesto_stats, lider_puesto_stats):
    """Export to Excel."""
    output_path = BASE_DIR / 'Analisis_Mesas_Probabilidad.xlsx'
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        lider_stats.to_excel(writer, sheet_name='Resumen Lideres', index=False)
        puesto_stats.to_excel(writer, sheet_name='Resumen Puestos', index=False)
        lider_puesto_stats.to_excel(writer, sheet_name='Lider x Puesto', index=False)
        result.to_excel(writer, sheet_name='Detalle Mesas', index=False)

        for sheet_name in writer.sheets:
            writer.sheets[sheet_name].set_column('A:Z', 18)

    print(f"Exportado: {output_path}", flush=True)


def generar_datos_dashboard(lider_stats, puesto_stats, result):
    """Generate JSON data for the dashboard."""
    # Top 20 lideres
    top_lideres = lider_stats.head(20).to_dict('records')
    for d in top_lideres:
        for k, v in d.items():
            if pd.isna(v):
                d[k] = 0

    # Top 30 puestos
    top_puestos = puesto_stats.head(30).to_dict('records')
    for d in top_puestos:
        for k, v in d.items():
            if pd.isna(v):
                d[k] = 0

    # Global stats
    result['mesa_id'] = result['Municipio'] + '_' + result['zona_code'].astype(str) + '_' + result['Lugar'] + '_' + result['mesa_num'].astype(str)
    total_simp = int(result['simpatizantes'].sum())
    total_mesas_presencia = int(result['mesa_id'].nunique())
    total_mesas_con_votos = int(result[result['tiene_votos'] == 1]['mesa_id'].nunique())
    votos_atribuidos = int(result['votos_atribuidos'].sum())
    prob_global = round(votos_atribuidos / max(total_simp, 1), 4)

    return {
        'lideres': top_lideres,
        'puestos': top_puestos,
        'global': {
            'total_simpatizantes': total_simp,
            'total_mesas_presencia': total_mesas_presencia,
            'total_mesas_con_votos': total_mesas_con_votos,
            'votos_atribuidos': votos_atribuidos,
            'prob_global': prob_global,
            'total_votos_candidata': TOTAL_VOTOS_CANDIDATA,
        }
    }


def main():
    print("=== Analisis de Probabilidad de Voto por Mesa ===", flush=True)

    print("\n1. Cargando datos...", flush=True)
    df_cam, df_sim_valle = load_data()

    print("\n2. Mapeando Lugar -> pto...", flush=True)
    lugar_pto_map = build_lugar_to_pto_map(df_cam, df_sim_valle)

    print("\n3. Analizando mesas...", flush=True)
    result = analisis_por_mesa(df_cam, df_sim_valle, lugar_pto_map)
    print(f"  {len(result):,} registros lider-puesto-mesa generados", flush=True)
    print(f"  Mesas con votos candidata: {result['tiene_votos'].sum():,} / {len(result):,} ({result['tiene_votos'].mean()*100:.1f}%)", flush=True)

    print("\n4. Generando resumenes...", flush=True)
    lider_stats = resumen_por_lider(result)
    puesto_stats = resumen_por_puesto(result)
    lider_puesto_stats = resumen_por_lider_puesto(result)

    print("\n5. Exportando Excel...", flush=True)
    exportar_excel(result, lider_stats, puesto_stats, lider_puesto_stats)

    print("\n6. Generando datos para dashboard...", flush=True)
    dashboard_data = generar_datos_dashboard(lider_stats, puesto_stats, result)

    # Save as JSON for dashboard use
    json_path = BASE_DIR / 'mesa_data.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dashboard_data, f, ensure_ascii=False, indent=2)
    print(f"  JSON: {json_path}", flush=True)

    # Print summary
    print("\n=== RESUMEN ===", flush=True)
    gs = dashboard_data['global']
    print(f"  Simpatizantes analizados: {gs['total_simpatizantes']:,}", flush=True)
    print(f"  Mesas con presencia: {gs['total_mesas_presencia']:,}", flush=True)
    print(f"  Mesas con votos candidata: {gs['total_mesas_con_votos']:,}", flush=True)
    print(f"  Probabilidad global de voto: {gs['prob_global']*100:.2f}%", flush=True)

    print("\n--- Top 10 Lideres por simpatizantes ---", flush=True)
    top10 = lider_stats.nlargest(10, 'simpatizantes')
    for _, r in top10.iterrows():
        prob = r['prob_voto'] if pd.notna(r['prob_voto']) else 0
        ratio = r['ratio_votos_simp'] if pd.notna(r['ratio_votos_simp']) else 0
        print(f"  {r['Líder']}: {r['simpatizantes']:,} simp | "
              f"{r['mesas_con_votos']}/{r['mesas_con_presencia']} mesas con votos ({r['pct_mesas_con_votos']:.0f}%) | "
              f"prob={prob*100:.1f}% | ratio={ratio:.2f}x", flush=True)

    print("\nDONE", flush=True)
    return dashboard_data


if __name__ == '__main__':
    main()
