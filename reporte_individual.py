"""
Reporte Individual de Simpatizantes - Analisis Forense
======================================================
Genera un listado por cedula de cada simpatizante con:
- Datos personales (cedula, nombre, lider)
- Puesto de votacion, mesa
- Votos reales en su mesa
- Calculo estadistico de probabilidad de voto
- Veredicto individual: CUMPLIO / PROBABLE / INDETERMINADO / NO CUMPLIO

Modelo estadistico:
  Para cada simpatizante en una mesa con V votos candidata,
  S_l simpatizantes del mismo lider y S_t simpatizantes totales:

  1. P(voto) = min(V, S_t) / S_t  (probabilidad base Bayesiana)
  2. Indice de certeza = f(exclusividad, densidad, excedente)
     - Exclusividad: el lider es el unico en esa mesa? (So == 0)
     - Densidad: ratio simpatizantes/votos → si V >> S_t hay apoyo externo
     - Cobertura: si V >= S_l y So=0, alta certeza para ese lider

  3. Veredicto:
     - CUMPLIO:        P >= 0.8 Y certeza >= 0.7  (evidencia fuerte)
     - PROBABLE:       P >= 0.5 O (V > 0 y certeza >= 0.5)
     - INDETERMINADO:  V > 0 pero P < 0.5 y certeza < 0.5 (compartida)
     - NO CUMPLIO:     V == 0 (sin ninguna duda)
"""

import pandas as pd
import numpy as np
import pickle, os, json
from pathlib import Path

from analisis_mesas import (
    CANDIDATA_CODE, PARTIDO_CODE, TOTAL_VOTOS_CANDIDATA,
    MPIO_MAP, MPIO_NAME_TO_CODE, BASE_DIR,
    load_data, build_lugar_to_pto_map
)


def construir_reporte_individual(df_cam, df_sim_valle, lugar_pto_map):
    """
    Construye un dataframe con UNA FILA POR SIMPATIZANTE (cedula)
    cruzado con datos electorales de su mesa.
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

    # Votos por zona+mesa (sumando todos los ptos — para mapeos no-exact)
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

    # --- Simpatizantes individuales con zona inferida ---
    sim = df_sim_valle.dropna(subset=['mesa_num']).copy()
    sim['mesa_num'] = sim['mesa_num'].astype(int)

    # --- Conteo de simpatizantes por mesa (del mismo lider y total) ---
    simp_por_lider_mesa = (
        sim.groupby(['Líder', 'mpio_code', 'zona_code', 'Lugar', 'mesa_num'])
        .agg(simp_lider=('Cédula', 'count'))
        .reset_index()
    )
    simp_total_mesa = (
        sim.groupby(['mpio_code', 'zona_code', 'Lugar', 'mesa_num'])
        .agg(simp_total=('Cédula', 'count'), n_lideres=('Líder', 'nunique'))
        .reset_index()
    )

    # --- Agregar conteos a cada simpatizante ---
    sim = sim.merge(
        simp_por_lider_mesa,
        on=['Líder', 'mpio_code', 'zona_code', 'Lugar', 'mesa_num'],
        how='left'
    )
    sim = sim.merge(
        simp_total_mesa,
        on=['mpio_code', 'zona_code', 'Lugar', 'mesa_num'],
        how='left'
    )
    sim['simp_otros'] = sim['simp_total'] - sim['simp_lider']

    # --- Mapeo Lugar -> pto desde catalogo ---
    sim = sim.merge(
        lugar_pto_map[['mpio', 'zona', 'lugar', 'pto', 'match_type']],
        left_on=['mpio_code', 'Lugar'],
        right_on=['mpio', 'lugar'],
        how='left'
    )
    # Usar zona del catalogo cuando el mapeo es confiable
    has_cat_zona = sim['match_type'].isin(['exact', 'fuzzy'])
    sim.loc[has_cat_zona, 'zona_code'] = sim.loc[has_cat_zona, 'zona']

    # --- GRUPO 1: Mapeo 'exact' o 'fuzzy' → cruzar por mpio+zona+pto+mesa ---
    has_pto = sim[sim['match_type'].isin(['exact', 'fuzzy'])].copy()
    has_pto['pto'] = has_pto['pto'].astype(int)

    result_matched = has_pto.merge(
        votos_mesa_pto,
        left_on=['mpio_code', 'zona_code', 'pto', 'mesa_num'],
        right_on=['mpio', 'zona', 'pto', 'mesa'],
        how='left',
        suffixes=('', '_elec')
    )

    # --- GRUPO 2: sin mapeo → cruzar por mpio+zona+mesa (suma ptos) ---
    no_match = sim[~sim['match_type'].isin(['exact', 'fuzzy'])].copy()

    result_zona = no_match.merge(
        votos_mesa_zona,
        left_on=['mpio_code', 'zona_code', 'mesa_num'],
        right_on=['mpio', 'zona', 'mesa'],
        how='left',
        suffixes=('', '_zona')
    )
    result_zona['match_type'] = 'zona_mesa'

    # --- Combinar ---
    cols = ['Cédula', 'Nombres', 'Apellidos', 'Líder', 'Categoría líder',
            'Municipio', 'zona_code', 'Lugar', 'mesa_num',
            'simp_lider', 'simp_otros', 'simp_total', 'n_lideres',
            'votos_candidata', 'votos_totales', 'match_type']

    for c in cols:
        if c not in result_matched.columns:
            result_matched[c] = None
        if c not in result_zona.columns:
            result_zona[c] = None

    result = pd.concat([
        result_matched[cols],
        result_zona[cols]
    ], ignore_index=True)

    result['votos_candidata'] = result['votos_candidata'].fillna(0).astype(int)
    result['votos_totales'] = result['votos_totales'].fillna(0).astype(int)
    result['simp_lider'] = result['simp_lider'].fillna(0).astype(int)
    result['simp_otros'] = result['simp_otros'].fillna(0).astype(int)
    result['simp_total'] = result['simp_total'].fillna(0).astype(int)
    result['n_lideres'] = result['n_lideres'].fillna(1).astype(int)

    return result


def calcular_probabilidad_voto(df):
    """
    Calcula la probabilidad estadistica de que cada simpatizante haya votado.

    Modelo:
    -------
    V  = votos_candidata en la mesa
    Sl = simpatizantes de ESTE lider en la mesa
    So = simpatizantes de OTROS lideres en la mesa
    St = Sl + So

    1. P_base = min(V, St) / St
       Probabilidad uniforme de que cualquier simpatizante en esa mesa haya votado.
       Si V >= St: P_base = 1.0 (suficientes votos para todos)
       Si V < St: P_base = V/St (no alcanzan, se distribuyen uniformemente)

    2. Indice de certeza (0 a 1):
       - exclusividad: 0.4 si So==0 (solo este lider), 0.0 si compartida
       - suficiencia:  0.3 si V >= Sl (votos suficientes para este lider), escalado si no
       - concentracion: 0.3 * min(Sl/St, 1.0) (que porcion de la mesa es de este lider)
       Total maximo = 1.0

    3. P_ajustada = P_base * (0.5 + 0.5 * certeza)
       La certeza modula la probabilidad: certeza alta sube P, certeza baja la baja.

    4. Veredicto:
       - NO CUMPLIO:     V == 0 → sin votos, certeza absoluta
       - CUMPLIO:        P_ajustada >= 0.75 Y certeza >= 0.6
       - PROBABLE:       P_ajustada >= 0.40 O (V > 0 y certeza >= 0.5)
       - INDETERMINADO:  hay votos pero no se puede atribuir con confianza
    """
    V = df['votos_candidata'].values
    Sl = df['simp_lider'].values
    So = df['simp_otros'].values
    St = df['simp_total'].values

    # 1. Probabilidad base
    St_safe = np.where(St > 0, St, 1)
    P_base = np.minimum(V, St) / St_safe

    # 2. Indice de certeza
    # a) Exclusividad: 0.4 si este lider es el unico en la mesa
    exclusividad = np.where(So == 0, 0.4, 0.0)

    # b) Suficiencia: 0.3 si V >= Sl, escala proporcional si no
    Sl_safe = np.where(Sl > 0, Sl, 1)
    suficiencia = 0.3 * np.minimum(V / Sl_safe, 1.0)

    # c) Concentracion: que porcion de la mesa pertenece a este lider
    concentracion = 0.3 * (Sl / St_safe)

    certeza = exclusividad + suficiencia + concentracion

    # 3. Probabilidad ajustada
    P_ajustada = P_base * (0.5 + 0.5 * certeza)

    # 4. Veredicto
    veredicto = np.full(len(df), 'INDETERMINADO', dtype=object)
    veredicto[V == 0] = 'NO CUMPLIO'
    veredicto[(P_ajustada >= 0.75) & (certeza >= 0.6) & (V > 0)] = 'CUMPLIO'
    veredicto[(veredicto == 'INDETERMINADO') &
              ((P_ajustada >= 0.40) | ((V > 0) & (certeza >= 0.5)))] = 'PROBABLE'

    # 5. Nivel de confianza del calculo
    nivel_confianza = np.full(len(df), 'BAJA', dtype=object)
    nivel_confianza[certeza >= 0.6] = 'ALTA'
    nivel_confianza[(certeza >= 0.35) & (certeza < 0.6)] = 'MEDIA'
    nivel_confianza[V == 0] = 'ALTA'  # certeza absoluta de que no voto

    df['prob_base'] = np.round(P_base, 4)
    df['certeza'] = np.round(certeza, 4)
    df['prob_ajustada'] = np.round(P_ajustada, 4)
    df['veredicto'] = veredicto
    df['nivel_confianza'] = nivel_confianza

    return df


def generar_resumen(df):
    """Genera resumen de veredictos por lider."""
    # Por lider
    resumen = df.groupby('Líder').agg(
        total_simp=('Cédula', 'count'),
        cumplio=('veredicto', lambda x: (x == 'CUMPLIO').sum()),
        probable=('veredicto', lambda x: (x == 'PROBABLE').sum()),
        indeterminado=('veredicto', lambda x: (x == 'INDETERMINADO').sum()),
        no_cumplio=('veredicto', lambda x: (x == 'NO CUMPLIO').sum()),
        prob_promedio=('prob_ajustada', 'mean'),
        certeza_promedio=('certeza', 'mean'),
        municipios=('Municipio', 'nunique'),
        puestos=('Lugar', 'nunique'),
    ).reset_index()

    resumen['%_cumplio'] = (resumen['cumplio'] / resumen['total_simp'] * 100).round(1)
    resumen['%_probable'] = (resumen['probable'] / resumen['total_simp'] * 100).round(1)
    resumen['%_no_cumplio'] = (resumen['no_cumplio'] / resumen['total_simp'] * 100).round(1)
    resumen['%_indeterminado'] = (resumen['indeterminado'] / resumen['total_simp'] * 100).round(1)
    resumen['prob_promedio'] = (resumen['prob_promedio'] * 100).round(1)
    resumen['certeza_promedio'] = (resumen['certeza_promedio'] * 100).round(1)

    # Veredicto global del lider
    resumen['veredicto_lider'] = 'NO CUMPLIO'
    total_positivo = resumen['cumplio'] + resumen['probable']
    pct_positivo = total_positivo / resumen['total_simp'] * 100
    resumen.loc[pct_positivo >= 60, 'veredicto_lider'] = 'CUMPLIO'
    resumen.loc[(pct_positivo >= 30) & (pct_positivo < 60), 'veredicto_lider'] = 'PARCIAL'

    return resumen.sort_values('total_simp', ascending=False)


def exportar_excel(df_individual, resumen_lider):
    """Export to Excel with formatting."""
    output_path = BASE_DIR / 'Reporte_Individual_Simpatizantes.xlsx'

    # Preparar dataframes para export
    # Individual: ordenar por lider, municipio, puesto, mesa
    indiv = df_individual[[
        'Cédula', 'Nombres', 'Apellidos', 'Líder', 'Categoría líder',
        'Municipio', 'Lugar', 'mesa_num',
        'simp_lider', 'simp_otros', 'simp_total', 'n_lideres',
        'votos_candidata', 'votos_totales',
        'prob_base', 'certeza', 'prob_ajustada',
        'veredicto', 'nivel_confianza', 'match_type'
    ]].copy()
    indiv.rename(columns={
        'mesa_num': 'Mesa',
        'simp_lider': 'Simp. Lider (mesa)',
        'simp_otros': 'Simp. Otros (mesa)',
        'simp_total': 'Simp. Total (mesa)',
        'n_lideres': 'Lideres (mesa)',
        'votos_candidata': 'Votos Candidata (mesa)',
        'votos_totales': 'Votos Totales (mesa)',
        'prob_base': 'Prob. Base',
        'certeza': 'Indice Certeza',
        'prob_ajustada': 'Prob. Ajustada',
        'veredicto': 'Veredicto',
        'nivel_confianza': 'Nivel Confianza',
        'match_type': 'Tipo Cruce'
    }, inplace=True)
    indiv = indiv.sort_values(['Líder', 'Municipio', 'Lugar', 'Mesa'])

    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        # Sheet 1: Resumen por lider
        resumen_lider.to_excel(writer, sheet_name='Resumen Lideres', index=False)

        # Sheet 2: Detalle individual
        indiv.to_excel(writer, sheet_name='Detalle Simpatizantes', index=False)

        # Sheet 3: Metodologia
        metodo = pd.DataFrame({
            'Concepto': [
                'Prob. Base',
                'Indice Certeza',
                'Prob. Ajustada',
                'Veredicto CUMPLIO',
                'Veredicto PROBABLE',
                'Veredicto INDETERMINADO',
                'Veredicto NO CUMPLIO',
                'Nivel Confianza ALTA',
                'Nivel Confianza MEDIA',
                'Nivel Confianza BAJA',
                'Exclusividad (Certeza)',
                'Suficiencia (Certeza)',
                'Concentracion (Certeza)',
            ],
            'Formula': [
                'min(Votos, Simp.Total) / Simp.Total',
                'Exclusividad + Suficiencia + Concentracion (0 a 1)',
                'Prob.Base * (0.5 + 0.5 * Certeza)',
                'Prob.Ajustada >= 0.75 Y Certeza >= 0.6',
                'Prob.Ajustada >= 0.40 O (hay votos Y Certeza >= 0.5)',
                'Hay votos pero no se puede atribuir con confianza',
                'Votos Candidata = 0 en la mesa (certeza total)',
                'Certeza >= 0.6 o V=0',
                'Certeza entre 0.35 y 0.6',
                'Certeza < 0.35',
                '0.4 si el lider es el unico en la mesa, 0 si compartida',
                '0.3 * min(V/Sl, 1.0)',
                '0.3 * (Sl/St)',
            ],
            'Descripcion': [
                'Probabilidad uniforme: si hay suficientes votos para todos los simp, P=1.0',
                'Indice que mide que tan confiable es atribuir el voto a este simpatizante',
                'Probabilidad corregida por la certeza (certeza alta sube P, baja la reduce)',
                'Evidencia FUERTE: pocos simpatizantes, votos suficientes, certeza alta',
                'Evidencia MODERADA: hay votos y la distribucion sugiere que pudo haber votado',
                'AMBIGUO: mesa compartida con muchos simp. de otros lideres, votos insuficientes',
                'DEFINITIVO: Cero votos en la mesa = no voto por la candidata',
                'La clasificacion es de alta confiabilidad estadistica',
                'Hay cierta ambiguedad pero la clasificacion es razonable',
                'La clasificacion es estimada, no se puede confirmar con certeza',
                'Si solo este lider tiene simpatizantes en la mesa, la atribucion es mas confiable',
                'Si los votos cubren al menos los simpatizantes de este lider, mayor certeza',
                'Si este lider tiene alta porcion de simpatizantes en la mesa, mejor atribucion',
            ]
        })
        metodo.to_excel(writer, sheet_name='Metodologia', index=False)

        wb = writer.book
        # Formats
        fmt_green = wb.add_format({'bg_color': '#c8e6c9', 'font_color': '#1b5e20'})
        fmt_red = wb.add_format({'bg_color': '#ffcdd2', 'font_color': '#b71c1c'})
        fmt_orange = wb.add_format({'bg_color': '#fff3e0', 'font_color': '#e65100'})
        fmt_gray = wb.add_format({'bg_color': '#f5f5f5', 'font_color': '#616161'})

        # Format Resumen
        ws_res = writer.sheets['Resumen Lideres']
        ws_res.set_column('A:A', 30)
        ws_res.set_column('B:Z', 14)
        vl_col = list(resumen_lider.columns).index('veredicto_lider')
        n = len(resumen_lider) + 1
        ws_res.conditional_format(1, vl_col, n, vl_col, {
            'type': 'cell', 'criteria': '==', 'value': '"CUMPLIO"', 'format': fmt_green
        })
        ws_res.conditional_format(1, vl_col, n, vl_col, {
            'type': 'cell', 'criteria': '==', 'value': '"NO CUMPLIO"', 'format': fmt_red
        })
        ws_res.conditional_format(1, vl_col, n, vl_col, {
            'type': 'cell', 'criteria': '==', 'value': '"PARCIAL"', 'format': fmt_orange
        })

        # Format individual
        ws_ind = writer.sheets['Detalle Simpatizantes']
        ws_ind.set_column('A:A', 14)  # cedula
        ws_ind.set_column('B:C', 16)  # nombres
        ws_ind.set_column('D:D', 30)  # lider
        ws_ind.set_column('E:E', 20)  # cat lider
        ws_ind.set_column('F:G', 20)  # mpio, lugar
        ws_ind.set_column('H:T', 14)  # numeric cols

        v_col = list(indiv.columns).index('Veredicto')
        n_ind = len(indiv) + 1
        ws_ind.conditional_format(1, v_col, n_ind, v_col, {
            'type': 'cell', 'criteria': '==', 'value': '"CUMPLIO"', 'format': fmt_green
        })
        ws_ind.conditional_format(1, v_col, n_ind, v_col, {
            'type': 'cell', 'criteria': '==', 'value': '"NO CUMPLIO"', 'format': fmt_red
        })
        ws_ind.conditional_format(1, v_col, n_ind, v_col, {
            'type': 'cell', 'criteria': '==', 'value': '"PROBABLE"', 'format': fmt_orange
        })
        ws_ind.conditional_format(1, v_col, n_ind, v_col, {
            'type': 'cell', 'criteria': '==', 'value': '"INDETERMINADO"', 'format': fmt_gray
        })

        # Metodologia
        ws_met = writer.sheets['Metodologia']
        ws_met.set_column('A:A', 28)
        ws_met.set_column('B:B', 55)
        ws_met.set_column('C:C', 80)

    print(f"  Exportado: {output_path}", flush=True)
    return output_path


def main():
    print("=" * 60, flush=True)
    print("  REPORTE INDIVIDUAL DE SIMPATIZANTES", flush=True)
    print("=" * 60, flush=True)

    print("\n1. Cargando datos...", flush=True)
    df_cam, df_sim_valle = load_data()

    print("\n2. Mapeando Lugar -> pto...", flush=True)
    lugar_pto_map = build_lugar_to_pto_map(df_cam, df_sim_valle)

    print("\n3. Construyendo reporte individual...", flush=True)
    df_indiv = construir_reporte_individual(df_cam, df_sim_valle, lugar_pto_map)
    print(f"  {len(df_indiv):,} simpatizantes con cruce electoral", flush=True)

    print("\n4. Calculando probabilidad de voto...", flush=True)
    df_indiv = calcular_probabilidad_voto(df_indiv)

    # Distribucion
    dist = df_indiv['veredicto'].value_counts()
    total = len(df_indiv)
    print("  Distribucion de veredictos:", flush=True)
    for v in ['CUMPLIO', 'PROBABLE', 'INDETERMINADO', 'NO CUMPLIO']:
        n = dist.get(v, 0)
        print(f"    {v}: {n:,} ({n/total*100:.1f}%)", flush=True)

    # Confianza
    conf = df_indiv['nivel_confianza'].value_counts()
    print("  Nivel de confianza:", flush=True)
    for c in ['ALTA', 'MEDIA', 'BAJA']:
        n = conf.get(c, 0)
        print(f"    {c}: {n:,} ({n/total*100:.1f}%)", flush=True)

    print("\n5. Generando resumen por lider...", flush=True)
    resumen = generar_resumen(df_indiv)

    print("\n6. Exportando Excel...", flush=True)
    exportar_excel(df_indiv, resumen)

    # Print top leaders
    print("\n" + "=" * 60, flush=True)
    print("  RESUMEN POR LIDER (Top 15)", flush=True)
    print("=" * 60, flush=True)
    for _, r in resumen.head(15).iterrows():
        print(
            f"  {r['Líder']}: {r['total_simp']:,} simp | "
            f"Cumplio={r['cumplio']} ({r['%_cumplio']}%) "
            f"Probable={r['probable']} ({r['%_probable']}%) "
            f"Indet.={r['indeterminado']} ({r['%_indeterminado']}%) "
            f"No={r['no_cumplio']} ({r['%_no_cumplio']}%) | "
            f"{r['veredicto_lider']}",
            flush=True
        )

    print("\nDONE", flush=True)


if __name__ == '__main__':
    main()
