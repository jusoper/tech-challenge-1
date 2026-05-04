# ML Canvas — Previsão de Churn (operadora de telecom)

Documento alinhado à fase de **Business Understanding** e **Data Understanding** (CRISP-DM), conforme *Ciclo de Vida dos Modelos — Aula 01* (`aula_01_entendendo-o-problema-de-negocio-e-os-dados.txt`).

## 1. Problema de negócio

A operadora perde clientes em ritmo acelerado. O objetivo é **priorizar clientes com maior probabilidade de cancelar** para ações de retenção (contato proativo, oferta, suporte), reduzindo churn e preservando receita.

**Hipótese de valor:** concentrar o esforço de retenção nos casos em que a intervenção ainda pode mudar o resultado (antes do cancelamento efetivo).

## 2. Stakeholders

| Papel | Interesse |
|--------|-----------|
| Diretoria / sponsor | Redução de churn, ROI das campanhas de retenção |
| CRM / Retenção | Lista priorizada, operação dentro de capacidade de contato |
| Marketing | Consistência de oferta, risco de irritar cliente “falso positivo” |
| TI / dados | Acesso a dados, LGPD, latência e disponibilidade da API |
| Cientista de dados | Definição de alvo, métricas, baseline e experimentação (MLflow) |

## 3. Decisão assistida por ML

**Tarefa:** classificação binária — para cada cliente ativo, estimar **risco de churn** (probabilidade ou score).

**Saída esperada:** score + eventual classe após **threshold** operacional (definido com negócio conforme custo FP vs FN).

## 4. Métricas de negócio (KPI)

| KPI | Definição inicial | Notas |
|-----|-------------------|--------|
| Taxa de churn | % clientes que cancelam no período | Acompanhar antes/depois das ações |
| **Custo de churn evitado** (proxy) | Soma de **LTV (ou margem)** dos clientes retidos graças à priorização | Exige premissas de LTV e taxa de sucesso da ação; usada como *métrica de negócio* no notebook |
| Custo por contato | R$/contato × volume | Limita quantos clientes podemos priorizar |

**SLOs / critérios de serviço (rascunho para evolução na Etapa 4)**

| SLO | Alvo inicial (exemplo) | Motivo |
|-----|------------------------|--------|
| Cobertura de scoring | ≥ 99% dos clientes elegíveis com predição diária | Operação de CRM |
| Latência API (p95) | &lt; 500 ms por requisição síncrona (a definir na Etapa 3) | Uso em tempo real vs batch |
| Drift | Alerta se PSI ou KS cruzar limiar (a calibrar) | Manutenção do modelo |

## 5. Métricas técnicas (alinhamento com negócio)

Definidas no notebook **01_eda_baselines** com base no roteiro do tech challenge e no material de **Fundamentos** (classificação: precisão/revocação/F1; curvas **ROC** e **PR** em *Aula 06* do extrato `aula_06_machine-learning-aula-06.txt`).

- **ROC-AUC:** capacidade de ordenar positivos vs negativos em vários limiares.  
- **PR-AUC (average precision):** sensível ao desbalanceamento — útil em churn.  
- **F1:** equilíbrio precisão–revocação em um limiar escolhido.

## 6. Dados e compliance

- Dataset público tabular **Telco Customer Churn (IBM)** — sem dados reais da empresa.  
- Em produção: consentimento, base legal, minimização, anonimização e controle de acesso (LGPD).

## 7. Riscos e premissas

- Churn histórico pode não refletir futuro (mudança de mercado, precificação).  
- LTV e custos de ação são **premissas** — sensibilidade deve ser discutida com stakeholders.  
- Variáveis faltantes ou categorias novas exigem pipeline robusto (Etapa 3).

## 8. Experimentação (MVP)

Conforme *Ciclo de Vida — Aula 02* (`aula_02_experimentacao-e-mvp-de-modelos.txt`): EDA, baselines **DummyClassifier** e **Regressão logística**, registro no **MLflow** (parâmetros, métricas, versão do dataset).
