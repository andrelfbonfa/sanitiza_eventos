# Análise de Alertas Ineficazes

Este projeto identifica alertas de monitoramento sem efetividade, baseado em técnicas inspiradas na Netflix e regras que eu entendo que sejam interessantes com minha experiencia no assunto.

## OBS:

Muito importante rodar esse script em ambiente seguro, de preferencia em env, devido a algumas particularidades da biblioteca do pandas, precisamos dela super atualizada e com algumas modificações, que muito provavelmente quebraria qualquer outra aplicação que utiliza a mesma lib!!!!!!!!!

## Funcionalidades

- Analisa alertas por host, item e descrição.
- Calcula frequência, escopo, genericidade (usando embeddings de texto).
- Identifica repetição >50% do período e alertas resolvidos muito rápido.
- Gera CSV com alertas ineficazes e motivos explicativos.

## Estrutura de pastas

- `data/`: Coloque aqui o CSV de alertas.
- `scripts/`: Script principal.
- `results/`: CSVs gerados.


## Como usar

```bash
# Criar virtualenv (opcional, recomendado)
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

# Instalar dependências
pip install -r requirements.txt

# Rodar análise
python scripts/events_fadigue.py




