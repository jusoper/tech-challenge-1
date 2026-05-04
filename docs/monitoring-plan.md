# Plano de monitoramento — churn Telco

Documento alinhado a *Ciclo de Vida dos Modelos — Aula 05*: **métricas** (o que medir), **alertas** (quando escalar) e **playbook de resposta** (o que fazer). Complementa o [ML Canvas](ml-canvas.md) (SLOs iniciais), a [arquitetura de deploy](deploy-architecture.md) (batch + API) e o [Model Card](model-card.md) (cenários de falha).

**Versão:** 1.0 (maio/2026)  
**Escopo:** operação de **scoring batch** e **API FastAPI**; treino/experimentos seguem no **MLflow** com revisão humana periódica.

---

## 1. Princípios

| Princípio | Aplicação |
|-----------|-----------|
| **SLI → SLO → alerta** | Cada alerta deve ligar-se a um **SLI** (indicador bruto) e a um **SLO** (compromisso); evitar ruído sem dono. |
| **Separar sintoma de causa** | Latência alta (sintoma) pode ser CPU, dependência ou modelo; playbook investiga em camadas. |
| **Negócio + ML + plataforma** | Métricas de produto (cobertura, churn) convivem com deriva e saúde da API. |

Limiares numéricos abaixo marcados como **“a calibrar”** devem ser ajustados com baseline de 2–4 semanas em ambiente de homologação ou pós go-live.

---

## 2. Métricas (o que coletar)

### 2.1 Pipeline batch (materialização de scores)

| ID | Métrica (SLI) | Descrição | SLO / meta inicial |
|----|----------------|-----------|---------------------|
| B1 | **Cobertura de scoring** | % de clientes elegíveis com registro de score na janela do dia | **≥ 99%** (ML Canvas) |
| B2 | **Sucesso do job** | Jobs concluídos com exit 0 / total agendado | 100% por dia útil; tolerância zero para falha silenciosa |
| B3 | **Atraso (freshness)** | `now − max(timestamp_ingestão_scores)` | Ex.: &lt; 6 h após janela oficial do DW (a calibrar) |
| B4 | **Volume processado** | Linhas lidas vs esperadas (reconciliação com cadastro) | Desvio &lt; 1% ou investigação obrigatória |
| B5 | **Distribuição do score** | Histograma / média / percentis da probabilidade | Baseline registrada; desvio brusco aciona revisão (ver §3) |
| B6 | **Drift de entrada (PSI/KS)** | PSI por feature contínua crítica ou KS vs baseline de treino | Alerta se **PSI &gt; 0,25** (regra comum; **a calibrar**) ou KS acima do limiar acordado (ML Canvas) |
| B7 | **Qualidade de schema** | % linhas rejeitadas na validação / % valores imputados por coluna | Limiares por coluna definidos com dados (a calibrar) |

### 2.2 API real-time (`/predict`, `/health`)

| ID | Métrica (SLI) | Descrição | SLO / meta inicial |
|----|----------------|-----------|---------------------|
| A1 | **Disponibilidade** | % de checks `/health` OK em janela de 5 min | ≥ 99,5% mensal (exemplo; a calibrar) |
| A2 | **Latência p95** | Tempo de resposta `/predict` | **&lt; 500 ms** por requisição síncrona (rascunho ML Canvas; ajustar por payload) |
| A3 | **Taxa de erro HTTP** | 5xx / total | &lt; 0,1% sustentado 15 min (a calibrar) |
| A4 | **Taxa 4xx (validação)** | 4xx / total | Pico pode ser esperado; tendência crescente → possível quebra de contrato upstream |
| A5 | **Throughput** | requisições/s | Capacidade planejada + autoscaling documentado no deploy |

Logs estruturados (Etapa 3) devem incluir **correlation id**, **model_version** / hash do artefato e **latência** para suportar A2–A5.

### 2.3 Qualidade do modelo (pós-deploy, com rótulo tardio)

| ID | Métrica | Descrição | Notas |
|----|---------|-----------|--------|
| M1 | **ROC-AUC / PR-AUC** em janela holdout temporal ou amostra rotulada | Desempenho discriminativo | Comparar com baseline do [Model Card](model-card.md); queda sustentada → alerta de conceito |
| M2 | **Estabilidade de calibragem** | Brier score ou curva de calibração por cohorte | Especialmente se decisões usarem probabilidade absoluta |
| M3 | **Taxa de positivos preditos** | % com score acima do limiar operacional | Desvio grande vs histórico pode indicar drift ou bug |

### 2.4 Negócio (proxy, com premissas do ML Canvas)

| ID | Métrica | Descrição |
|----|---------|-----------|
| N1 | **Taxa de churn** (cohortes priorizados vs controle) | Efetividade indireta das ações; requer desenho experimental |
| N2 | **Custo de churn evitado** (proxy) | Depende de LTV e taxa de sucesso da ação — revisão trimestral |

N1–N2 **não** substituem alertas técnicos: são **lag** altos e confundidos por campanhas; úteis em comitê de modelo.

---

## 3. Alertas (condições e severidade)

### 3.1 Matriz sugerida

| Condição | Severidade | Dono típico |
|----------|------------|-------------|
| Cobertura B1 &lt; 99% | **P1** | Engenharia de dados + ML Ops |
| Job batch falhou (B2) | **P1** | ML Ops |
| Freshness B3 acima do SLO | **P2** | Engenharia de dados |
| Desvio de volume B4 &gt; 1% | **P2** | Engenharia de dados |
| PSI/KS (B6) acima do limiar | **P2** (sustentado 2 janelas) | Ciência de dados |
| p95 latência A2 &gt; SLO por 15 min | **P2** | Plataforma / SRE |
| Taxa 5xx A3 acima do limiar | **P1** se &gt; 1%; senão **P2** | SRE |
| Queda M1 (AUC/PR-AUC) vs baseline acordada | **P2** → **P1** se negócio exposto | Ciência de dados + sponsor |

**Canais:** incidente P1 → página + canal de guerra; P2 → ticket + dashboard + revisão no comitê semanal de ML.

### 3.2 Anti-padrões

- Alerta sem runbook (ver §4).  
- Mesmo limiar em dev e prod sem ajuste de volume.  
- Apenas “acurácia” em churn desbalanceado — preferir **PR-AUC** e taxa de positivos (M3).

---

## 4. Playbook de resposta

### 4.1 Cobertura de scoring abaixo do SLO (B1)

1. Confirmar se o job rodou e se publicou na tabela destino.  
2. Checar filtros de elegibilidade (mudança de regra de negócio?).  
3. Validar join com cadastro de clientes (chaves, duplicatas).  
4. Comunicar CRM com ETA e, se necessário, **usar último score válido** com flag de staleness.  
5. Post-mortem: teste de reconciliação no CI do pipeline.

### 4.2 Job batch com falha (B2)

1. Inspecionar logs do orquestrador e stack trace da etapa que falhou.  
2. Reexecutar com **mesma versão** do pacote/artefato (reprodutibilidade).  
3. Se falha for dados: acionar §4.4.  
4. Se falha for memória/timeout: aumentar recursos ou particionar batch.  
5. Registrar incidente com ID do artefato e versão do schema.

### 4.3 Drift de dados (B6) ou queda de métricas (M1)

1. Gerar relatório PSI/KS por feature e cohorte (região, canal).  
2. Comparar distribuição de scores (B5) com baseline salva no MLflow.  
3. **Não** retreinar automaticamente sem aprovação: abrir experimento, validar em holdout temporal.  
4. Atualizar [Model Card](model-card.md) se limitações ou população mudarem.  
5. Se risco imediato: considerar **rollback** do artefato na API e limiar conservador temporário (menos FP ou menos FN — alinhar com negócio).

### 4.4 Erros de schema / validação (B7, A4)

1. Identificar campo que estourou validação (Pydantic/Pandera).  
2. Comparar contrato de dados com versão documentada da API.  
3. Corrigir upstream ou publicar **nova versão** da API com compatibilidade retroativa.  
4. DLQ ou bucket de “bad records” para reprocessamento.

### 4.5 Latência ou 5xx na API (A2, A3)

1. Verificar saúde de dependências (DB, se houver) e saturation de CPU.  
2. Confirmar que o processo carregou o artefato correto (`TELCO_SKLEARN_PIPELINE_PATH`).  
3. Escalar horizontalmente ou reduzir concorrência por instância conforme desenho.  
4. Se ataque ou pico anômalo: rate limit + WAF; ver [Model Card](model-card.md) §7.

### 4.6 Artefato errado ou fallback sintético em produção

1. Health check deve expor **identidade do modelo** (versão, hash, origem `joblib_file` vs `default_synthetic`).  
2. Se `default_synthetic` em produção: **P1** — trocar imediatamente para artefato aprovado ou degradar tráfego.  
3. Auditoria de variáveis de ambiente e pipeline de deploy.

---

## 5. Cadência de revisão

| Frequência | Atividade |
|------------|-----------|
| Diária | Painel batch B1–B5; fila de alertas P2 |
| Semanal | Drift B6 amostral; revisão A2–A4; backlog de schema |
| Mensal | Métricas M1–M3 em dados rotulados; comitê de modelo |
| Trimestral | N1–N2, SLOs e limiares “a calibrar”; atualização deste plano |

---

## 6. Referências internas

- SLOs iniciais: [ML Canvas §4](ml-canvas.md).  
- Padrões de serviço batch vs API: [deploy-architecture.md](deploy-architecture.md).  
- Riscos e falhas: [Model Card §7](model-card.md).

Atualizar este plano quando mudarem **definição de elegibilidade**, **artefato servido**, **fornecedor de observabilidade** ou **acordos de severidade** com o negócio.
