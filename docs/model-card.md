# Model Card — Previsão de churn (IBM Telco Customer Churn)

Documento alinhado a *Ciclo de Vida dos Modelos — Aula 03* (Model Card: performance, limitações, vieses e cenários de falha). Complementa o [ML Canvas](ml-canvas.md).

**Versão do documento:** 1.0 (maio/2026)  
**Pacote:** `telco_churn` (`pyproject.toml`)  
**Alvo:** `Churn` binário (1 = cliente cancelou no período observado).

---

## 1. Visão geral

| Item | Descrição |
|------|-----------|
| **Tarefa** | Classificação binária: estimar probabilidade de churn por cliente. |
| **Famílias de modelo** | Baselines Scikit-learn (dummy estratificado, regressão logística balanceada, Random Forest, HistGradientBoosting) e **MLP** (`ChurnMLP`) em PyTorch, conforme `telco_churn.evaluation.holdout.compare_models_holdout`. |
| **Saída principal** | Probabilidade da classe positiva (churn); decisão binária depende de **limiar** operacional (padrão de métricas no código: 0,5 para F1 e acurácia). |
| **API FastAPI** | Por padrão (sem artefatos em disco) sobe uma **MLP** treinada em dados sintéticos (`default_synthetic_mlp`). Em produção: definir **`TELCO_MLP_BUNDLE_PATH`** com `TelcoMlpPredictor` (prep + `ChurnMLP` alinhados ao treino). Opcional: **`TELCO_SKLEARN_PIPELINE_PATH`** para servir só sklearn. O melhor desempenho da tabela de pesquisa continua dependendo do experimento / MLflow — versionar o artefato servido. |

---

## 2. Dados de treino e avaliação

| Atributo | Detalhe |
|----------|---------|
| **Fonte** | IBM Telco Customer Churn (público), obtido via `scripts/download_data.py`; CSV em `data/raw/` (não versionado). |
| **Granularidade** | Uma linha por cliente; variáveis tabulares (serviços, contrato, cobranças, demografia simplificada). |
| **Pré-processamento** | `TelcoTableSanitizer`, imputação, `StandardScaler` em numéricos, `OneHotEncoder(handle_unknown="ignore")` em categóricas (`prepare_telco_features`, `build_telco_feature_transform_pipeline`). |
| **Split reportado** | Holdout **estratificado** 80/20 (`compare_models_holdout`) e, para estimativa com menor viés de amostra única, **CV estratificada k-fold OOF** (`compare_models_stratified_cv` / `compare_sklearn_baselines_stratified_cv`) com `StratifiedKFold` (padrão: 5 folds, `shuffle=True`, `random_state=42`), alinhado ao notebook Etapa 1. |
| **Desbalanceamento** | No snapshot avaliado: **n = 7 043**, taxa de churn **≈ 26,5%** (classe positiva minoritária). |

---

## 3. Desempenho (validação — holdout)

Métricas produzidas por `telco_churn.evaluation.metrics.compute_binary_metrics`: **ROC-AUC**, **PR-AUC** (average precision), **F1** e **acurácia**. F1 e acurácia usam **limiar 0,5** sobre a probabilidade predita.

**Resultados no conjunto de validação (20%) com seed 42** — execução reprodutível via código do pacote sobre o CSV local:

| Modelo | ROC-AUC | PR-AUC | F1 (t=0,5) | Acurácia (t=0,5) |
|--------|---------|--------|------------|------------------|
| dummy_stratified | 0,516 | 0,272 | 0,290 | 0,622 |
| logistic_regression_balanced | 0,841 | 0,632 | 0,613 | 0,737 |
| random_forest | 0,839 | 0,641 | 0,624 | 0,769 |
| hist_gradient_boosting | 0,835 | 0,646 | 0,618 | 0,753 |
| **churn_mlp** | **0,844** | **0,639** | **0,596** | **0,798** |

**Leitura:** a MLP apresenta **ROC-AUC** entre os melhores do conjunto avaliado; **PR-AUC** e **F1** variam com o limiar — para churn desbalanceado, **PR-AUC** e análise de custo FP/FN (`telco_churn.business.cost_tradeoff`) costumam ser mais alinhadas ao negócio do que acurácia isolada. Números exatos de novas execuções devem ser conferidos no **MLflow** (`mlruns/`, parâmetros e métricas por run).

**Treino MLP (hiperparâmetros padrão):** `TrainConfig` — AdamW, `BCEWithLogitsLoss`, batching, early stopping na loss de validação; arquitetura padrão `ChurnMLP(hidden_dims=(64, 32), dropout=0.1, activation="relu")` sobre vetor de features **após** o mesmo pré-processamento sklearn usado nos baselines.

---

## 4. Uso pretendido e não pretendido

**Pretendido:** priorização de carteira para **ações de retenção** (ordenação por risco), análises offline, protótipos de API e laboratório acadêmico (FIAP Tech Challenge).

**Não pretendido:** decisão automática sem supervisão humana; scoring de indivíduos fora do contexto de telecom contratual sem validação legal; substituição de políticas de crédito ou risco sem governança; inferência causal (“este cliente cancelou *porque* o modelo previu”).

---

## 5. Limitações

1. **Generalização temporal e de mercado:** o dataset é **histórico** e de contexto específico; mudanças de oferta, concorrência ou regulamentação degradam o modelo sem re-treino e monitoramento.  
2. **Proxy de valor:** métricas técnicas não incorporam sozinhas **LTV**, custo de contato nem taxa de sucesso da ação — exigem camada de negócio (ver ML Canvas).  
3. **Variância amostral:** holdout 80/20 e métricas OOF por k-fold respondem a perguntas ligeiramente diferentes; nenhum substitui conjunto de **produção** com janela temporal nem drift observado.  
4. **Variáveis não observadas:** fatores comportamentais, qualidade de rede ou insatisfação qualitativa podem não estar nas features.  
5. **Limiar fixo 0,5:** é convenção para F1/acurácia no código; o limiar operacional deve ser escolhido com **matriz de custo** (FP vs FN).  
6. **API vs experimento de pesquisa:** sem `TELCO_MLP_BUNDLE_PATH`, a API usa MLP **sintética** (não reflete o CSV completo). Desempenho em Telco real exige bundle ou pipeline sklearn versionado e testado de ponta a ponta.

---

## 6. Vieses, equidade e dados sensíveis

- **Viés de seleção e histórico:** clientes no dataset não são amostra aleatória de “todos os mercados”; padrões culturais, regionais ou de produto podem estar embutidos.  
- **Atributos sensíveis:** o IBM Telco inclui campos demográficos (ex.: `gender`). Uso em produção exige **base legal**, minimização de dados, avaliação de impacto diferencial em subgrupos e possível **remoção ou agregação** de atributos conforme LGPD e política interna.  
- **Rotulagem:** “churn” reflete definição operacional do dataset; mudança de definição (ex.: janela temporal) altera o alvo e invalida comparações antigas.  
- **Desbalanceamento:** modelos podem favorecer a classe majoritária em alguns limiares; métricas globais podem **mascarar** desempenho ruim em subpopulações — recomenda-se auditoria por segmento quando houver volume suficiente e governança.

---

## 7. Cenários de falha

| Cenário | Manifestação | Mitigação sugerida |
|--------|---------------|---------------------|
| **Deriva de dados (data drift)** | Queda de ROC-AUC/PR-AUC, PSI/KS fora do limiar | Retreino calendarizado, features estáveis, comparação de distribuições. |
| **Categorias novas / schema quebrado** | Erros de validação (Pandera/Pydantic) ou OHE com massa em “unknown” | Contrato de dados, versão de schema, fila DLQ, fallback seguro. |
| **Valores fora de domínio** | Inputs inválidos na API | Validação na borda, rejeição com código 4xx e logging estruturado. |
| **Indisponibilidade do artefato** | `TELCO_MLP_BUNDLE_PATH` / `TELCO_SKLEARN_PIPELINE_PATH` inválidos → MLP sintética | Alertas, health check, implantação imutável de bundle versionado. |
| **Ataque / carga anormal** | Latência alta ou erros em cascata | Rate limiting, autenticação, SLO de latência (ver ML Canvas). |
| **Expectativa de causalidade** | Negócio interpreta score como “causa” | Documentação e treinamento; escopo claramente preditivo. |

---

## 8. Reprodutibilidade e rastreio

- **Seeds:** `TrainConfig.seed` padrão 42; baselines com `random_state` fixo onde aplicável.  
- **Experimentos:** registro via MLflow (`log_compare_models_to_mlflow`) — parâmetros, métricas, curvas da MLP e artefatos.  
- **Testes:** `pytest` (smoke, schema, API) conforme `tests/`.

---

## 9. Contato e manutenção

Manutenção do modelo e deste card: equipe do repositório (Tech Challenge FIAP). Atualizar este documento quando mudarem **definição de alvo**, **pipeline**, **artefato servido na API** ou **resultados oficiais** de validação.
