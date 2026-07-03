
# 🍅 Detector Inteligente de Tomates utilizando YOLO11 e CNN
Aluno: Artur Romão  
Sistema de Visão Computacional desenvolvido em **Python** para detectar tomates em imagens e vídeos utilizando **YOLO11** e, posteriormente, classificar automaticamente seu estado de conservação utilizando uma **Rede Neural Convolucional (CNN)**.

O projeto foi desenvolvido como trabalho final da disciplina de **Visualização Computacional**, tendo como principal objetivo aplicar, na prática, todas as etapas de construção de um sistema de Inteligência Artificial: desde a coleta e análise dos dados até o treinamento dos modelos, avaliação, integração e execução em tempo real utilizando webcam.  

OBS:Para rodar os scripts vc deve baixar o dataset do kaggle e colocar numa pasta chamada dataset na origem.  
OBS: Foi usado o Python 3.12.10..  
OBS: Os modelos do YOLO treinados com as 300 imagens anotadas(CVAT) e os resultados estão na pasta \run.    
OBS: O dataset_yolo para treinar o YOLO e as labels foram disponibilizadas.  
OBS: Os modelos CNN treinado não foi disponibilizado pois é pesado de mais para o GIT,mas o script foi, com o dataset é só rodar o script de treino do cnn.  

---

# 📖 Introdução

A proposta inicial deste projeto era desenvolver um sistema capaz de responder automaticamente à seguinte pergunta:

> **O tomate está maduro, verde, velho ou estragado?**

Para isso, optei por dividir o problema em duas tarefas independentes.

A primeira consiste em **localizar o tomate** na imagem. Essa tarefa é conhecida como **Detecção de Objetos (Object Detection)** e foi realizada utilizando o modelo **YOLO11**, especializado em encontrar objetos em imagens e vídeos em tempo real.

Depois que o tomate é localizado, apenas a região correspondente ao fruto é recortada e enviada para um segundo modelo de Inteligência Artificial.

Esse segundo modelo é uma **Rede Neural Convolucional (CNN)** treinada exclusivamente para identificar o estado de conservação do tomate, classificando-o em uma das quatro categorias disponíveis:

- 🍅 Ripe (Maduro)
- 🟢 Unripe (Verde)
- ⚫ Old (Velho)
- ❌ Damaged (Estragado)

Essa abordagem apresenta uma vantagem importante: cada modelo é responsável por resolver apenas um problema específico.

Enquanto o YOLO aprende apenas **onde o tomate está**, a CNN aprende apenas **qual é o estado do tomate**.

Essa divisão torna o pipeline mais organizado, modular e fácil de manter.

Ao final do projeto, os dois modelos trabalham juntos em tempo real através da webcam, detectando automaticamente os tomates presentes na cena e informando sua classificação juntamente com o nível de confiança da previsão.

---

# 🎯 Objetivos

Os principais objetivos deste projeto foram:

- Desenvolver um pipeline completo de Visão Computacional.
- Treinar uma CNN para classificar tomates em quatro categorias.
- Treinar um modelo YOLO11 para detectar tomates em imagens.
- Integrar os dois modelos em um único sistema funcionando em tempo real.
- Aplicar conceitos de preparação de dados, treinamento, validação e avaliação de modelos.
- Explorar ferramentas utilizadas no desenvolvimento de aplicações modernas de Inteligência Artificial.

Além do resultado final, o foco do projeto foi compreender todas as etapas envolvidas na criação de um modelo de aprendizado profundo, incluindo as dificuldades encontradas durante o desenvolvimento.

---

# 🛠️ Tecnologias Utilizadas

Durante o desenvolvimento foram utilizadas diversas bibliotecas e ferramentas do ecossistema Python.

| Tecnologia | Finalidade |
|------------|------------|
| Python | Linguagem principal do projeto |
| PyTorch | Desenvolvimento e treinamento da CNN |
| Torchvision | Transformações das imagens e carregamento do dataset |
| OpenCV | Manipulação de imagens e captura da webcam |
| Ultralytics YOLO11 | Treinamento e inferência do modelo de detecção |
| NumPy | Operações numéricas |
| Matplotlib | Visualização de imagens e resultados |
| Scikit-Learn | Métricas auxiliares |
| Kaggle | Fonte do dataset |
| CVAT | Anotação das imagens para treinamento do YOLO |
| Docker Desktop | Execução local do CVAT |

---

# 📂 Estrutura do Projeto

```text
Tomato-Detection/
│
├── dataset/
│   ├── train/
│   └── val/
│
├── dataset_yolo/
│   ├── images/
│   ├── labels/
│   └── data.yaml
│
├── imagens_teste/
│
├── models/
│   └── tomato_cnn.pth
│
├── runs/
│
├── src/
│   ├── explorar_dataset.py
│   ├── visualizar_dataset.py
│   ├── train_cnn.py
│   ├── evaluate.py
│   ├── predict_image.py
│   ├── split_dataset.py
│   ├── train_yolo.py
│   ├── predict_yolo.py
│   ├── webcam_yolo.py
│   └── webcam_yolo_cnn.py
│
├── requirements.txt
└── README.md
```

---

# 📜 Scripts Desenvolvidos

A maior parte do projeto foi organizada dentro da pasta **src/**.

Cada script possui uma responsabilidade específica durante o pipeline de desenvolvimento.

| Script | Descrição |
|---------|------------|
| **src/explorar_dataset.py** | Analisa o dataset e apresenta estatísticas das classes. |
| **src/visualizar_dataset.py** | Permite visualizar imagens do dataset para inspeção visual. |
| **src/train_cnn.py** | Responsável pelo treinamento da Rede Neural Convolucional. |
| **src/evaluate.py** | Avalia o desempenho da CNN utilizando o conjunto de validação. |
| **src/predict_image.py** | Classifica imagens individuais utilizando a CNN treinada. |
| **src/split_dataset.py** | Divide automaticamente o dataset anotado em treino e validação para o YOLO. |
| **src/train_yolo.py** | Treina o modelo YOLO11 para detecção de tomates. |
| **src/predict_yolo.py** | Testa o YOLO utilizando imagens. |
| **src/webcam_yolo.py** | Executa a detecção em tempo real utilizando a webcam. |
| **src/webcam_yolo_cnn.py** | Integra YOLO e CNN em um único pipeline de detecção e classificação em tempo real. |

---

# ⚙️ Pipeline do Projeto

O funcionamento do sistema pode ser resumido pelo fluxo abaixo.

```text
Imagem ou Webcam
        │
        ▼
+----------------------+
|      YOLO11          |
| Detecta o tomate     |
+----------------------+
        │
        ▼
Recorta apenas o tomate
        │
        ▼
+----------------------+
|        CNN           |
| Classifica o tomate  |
+----------------------+
        │
        ▼
┌─────────────────────┐
│ Damaged             │
│ Old                 │
│ Ripe                │
│ Unripe              │
└─────────────────────┘
```

A utilização de dois modelos independentes torna o sistema mais modular.

O YOLO é responsável apenas por localizar os tomates na imagem, enquanto a CNN realiza exclusivamente a classificação do estado de conservação. Essa separação simplifica o treinamento de cada modelo e facilita futuras melhorias, permitindo substituir ou atualizar apenas uma das etapas sem alterar todo o pipeline.

---

# 🍅 Dataset

Para o treinamento dos modelos foi utilizado o seguinte dataset público disponível no Kaggle:

**Tomatoes Dataset**

https://www.kaggle.com/datasets/enalis/tomatoes-dataset

O conjunto de dados já disponibiliza as imagens separadas em dois grupos:

- **Train** (Treinamento)
- **Validation** (Validação)

Cada imagem pertence a uma das quatro classes:

| Classe | Significado |
|---------|-------------|
| Damaged | Tomate estragado |
| Old | Tomate velho |
| Ripe | Tomate maduro |
| Unripe | Tomate verde |

As imagens possuem resolução de **256 × 256 pixels**, facilitando o treinamento da CNN sem necessidade de grandes adaptações.

## Distribuição das imagens

### Treinamento

| Classe | Quantidade |
|---------|-----------:|
| Damaged | 941 |
| Old | 1992 |
| Ripe | 1967 |
| Unripe | 1585 |

### Validação

| Classe | Quantidade |
|---------|-----------:|
| Damaged | 106 |
| Old | 222 |
| Ripe | 220 |
| Unripe | 177 |

Ao todo, o dataset possui **~7.220 imagens**, sendo:

- **6.485 imagens para treinamento**
- **725 imagens para validação**

As imagens foram armazenadas localmente na pasta:

```text
dataset/
├── train/
└── val/
```

---

## 📊 Exploração do Dataset

O primeiro passo após obter o dataset foi analisar sua estrutura e verificar se os dados eram adequados para o treinamento.

Para isso foi desenvolvido o script:

> **📄 `src/explorar_dataset.py`**

Esse script realiza automaticamente:

- contagem das classes;
- verificação da distribuição do dataset;
- resolução das imagens;


Essa etapa foi importante para compreender possíveis limitações dos dados antes do treinamento dos modelos.

---

# 🔍 Análise Exploratória

Depois da análise quantitativa, foi realizada uma inspeção visual do dataset utilizando o script:

> **📄 `src/visualizar_dataset.py`**

Esse script permite visualizar exemplos das imagens pertencentes a cada classe apenas alterando o valor da variável `classe`.

Essa inspeção revelou diversas características importantes do conjunto de dados.

## ✔️ Pontos positivos

O dataset apresenta uma boa diversidade de:

- iluminação;
- posição dos tomates;
- ângulo de captura;
- orientação dos frutos;
- pequenas variações de fundo.

Essa diversidade ajuda a CNN a aprender características mais robustas e reduz a dependência de uma posição específica do tomate na imagem.

Outro ponto positivo é que o conjunto de treinamento possui mais de seis mil imagens, quantidade suficiente para treinar um primeiro modelo de classificação com resultados satisfatórios.

---

## ⚠️ Limitações encontradas

Apesar da boa quantidade de imagens, algumas limitações do dataset ficaram evidentes.

### 1. Desbalanceamento das classes

Embora relativamente equilibrado, o conjunto de treinamento possui menos exemplos da classe **Damaged**.

Distribuição aproximada:

- Damaged → **14%**
- Old → **30%**
- Ripe → **30%**
- Unripe → **24%**

Na prática, isso não causou grandes problemas durante o treinamento.

---

### 2. Sobreposição entre classes

A principal dificuldade encontrada está relacionada à qualidade da rotulação.

Algumas imagens classificadas como **Old** apresentam aparência muito semelhante a tomates maduros (**Ripe**).

Da mesma forma, algumas imagens classificadas como **Unripe** apresentam sinais de deterioração, tornando a separação visual entre as categorias bastante difícil.

Essa característica provavelmente explica por que a CNN apresentou maior dificuldade para distinguir tomates da classe **Old**, mesmo alcançando uma boa acurácia durante o treinamento.

---

### 3. Qualidade das imagens

Outra limitação observada foi a existência de imagens com baixa qualidade.

Algumas fotografias apresentam:

- baixa resolução;
- pouca nitidez;
- iluminação inadequada.

Esses fatores dificultam a extração de características importantes pelo modelo.

---

### 4. Viés do Dataset

Durante os testes foi possível observar outro comportamento importante.

Grande parte das imagens do dataset corresponde a **tomates italianos**.

Como consequência, tanto a CNN quanto o YOLO tendem a apresentar melhor desempenho quando recebem imagens desse mesmo tipo de tomate.

Esse comportamento evidencia um fenômeno comum em Aprendizado de Máquina: o modelo aprende melhor padrões presentes com maior frequência durante o treinamento.

---

## Considerações

Mesmo apresentando algumas limitações, o dataset mostrou-se adequado para fins acadêmicos.

As dificuldades encontradas refletem problemas reais enfrentados em projetos de Visão Computacional, nos quais a qualidade e a diversidade dos dados possuem impacto direto no desempenho final dos modelos.

Toda essa etapa de exploração foi fundamental para compreender as características do conjunto de dados antes do início do treinamento da CNN e do YOLO.

# 🧠 Treinamento da Rede Neural Convolucional (CNN)

Após analisar o dataset, a próxima etapa do projeto foi desenvolver um modelo capaz de identificar automaticamente o estado de conservação dos tomates.

Como o objetivo era classificar uma imagem em apenas uma das quatro categorias disponíveis, a arquitetura escolhida foi uma **Rede Neural Convolucional (Convolutional Neural Network - CNN)**.

As CNNs são atualmente uma das principais arquiteturas utilizadas em problemas de classificação de imagens, pois conseguem aprender automaticamente características visuais importantes, como bordas, texturas, formas e padrões, dispensando a necessidade de criar essas características manualmente.

Neste projeto, a CNN foi responsável por classificar os tomates nas seguintes categorias:

- 🍅 Ripe (Maduro)
- 🟢 Unripe (Verde)
- ⚫ Old (Velho)
- ❌ Damaged (Estragado)

---

# Por que utilizar uma CNN?

Diversos modelos poderiam ser utilizados para essa tarefa, porém a CNN apresentou algumas vantagens importantes.

## 1. Extração automática de características (Feature Extraction)

Uma das maiores vantagens da CNN é que ela aprende sozinha quais características são importantes durante o treinamento.

Enquanto técnicas tradicionais exigem que o desenvolvedor extraia manualmente informações como cor, textura ou formato, a CNN aprende essas representações automaticamente.

Durante o treinamento ocorre aproximadamente o seguinte processo:

- **Camadas iniciais:** aprendem bordas, linhas e contrastes;
- **Camadas intermediárias:** combinam essas informações para identificar formas e texturas;
- **Camadas finais:** reconhecem objetos mais complexos, como folhas, cascas e diferentes estados do tomate.

Esse processo permite que o modelo generalize melhor para imagens nunca vistas anteriormente.

---

## 2. Invariância espacial

Outra característica importante das CNNs é sua capacidade de reconhecer um objeto independentemente de sua posição na imagem.

Isso significa que, caso o modelo aprenda a identificar um tomate localizado no canto superior esquerdo da imagem, ele também será capaz de reconhecer esse mesmo tomate no centro ou em qualquer outra região da fotografia.

Essa propriedade torna a CNN extremamente eficiente para tarefas de classificação de imagens.

---

## 3. Baixo custo computacional

Outro fator que influenciou a escolha foi o hardware disponível.

Como o projeto foi desenvolvido utilizando um computador pessoal, optei por construir uma CNN própria, mais simples e leve, capaz de ser treinada em aproximadamente **1 hora**, mantendo um desempenho satisfatório.

---

# 📄 Script principal: `src/train_cnn.py`

Todo o treinamento da Rede Neural Convolucional foi realizado através do script:

```text
src/train_cnn.py
```

É responsável por transformar milhares de imagens em um modelo capaz de realizar classificações automaticamente.

Durante sua execução, o script realiza todas as etapas necessárias para o treinamento da CNN.

---

# 1. Preparação dos dados

A primeira etapa consiste em preparar o dataset para o treinamento.

O script utiliza o **PyTorch DataLoader**, responsável por carregar as imagens de forma eficiente.

Durante essa etapa são realizadas algumas transformações importantes.

## Redimensionamento

Todas as imagens são redimensionadas para:

```text
256 × 256 pixels
```

Esse é o tamanho esperado pela arquitetura da rede.

---

## Conversão para Tensor

Depois do redimensionamento, as imagens deixam de ser matrizes convencionais e passam a ser convertidas para **Tensors**, estrutura utilizada internamente pelo PyTorch.

---

## Organização em lotes (Batches)

Em vez de carregar todas as imagens na memória simultaneamente, o DataLoader divide o dataset em pequenos grupos.

No projeto foi utilizado:

```python
BATCH_SIZE = 32
```

Ou seja, a rede processa 32 imagens por vez.

Isso reduz o consumo de memória e acelera o treinamento.

---

## Embaralhamento das imagens

Durante o treinamento também é utilizado:

```python
shuffle=True
```

Assim, a ordem das imagens muda a cada época.

Isso evita que a rede memorize a sequência das imagens presentes nas pastas do dataset.

---

# 2. Arquitetura da CNN

A arquitetura desenvolvida foi dividida em duas partes principais.

## Camadas de Extração de Características (`features`)

A primeira parte da rede possui camadas convolucionais responsáveis por extrair informações importantes da imagem.

Essas camadas utilizam:

- Convoluções (`Conv2D`)
- Função de ativação ReLU
- Camadas MaxPooling

Ao longo dessas operações a rede aprende automaticamente características como:

- bordas;
- texturas;
- formatos;
- padrões visuais presentes nos tomates.

Cada camada aumenta o nível de abstração das informações aprendidas.

---

## Camadas de Classificação (`classifier`)

Após a extração das características, o mapa de características produzido pela CNN é transformado em um vetor unidimensional através da operação **Flatten**.

Em seguida, esse vetor passa por camadas totalmente conectadas (**Fully Connected Layers**), responsáveis pela classificação final.

A última camada possui:

```python
nn.Linear(512, 4)
```

Como existem quatro possíveis classes:

- Damaged
- Old
- Ripe
- Unripe

o modelo retorna um valor para cada uma delas.

A classe com maior probabilidade será considerada a resposta final.

---

# 3. Função de perda e Otimizador

Para que a rede consiga aprender, é necessário medir o erro cometido em cada previsão.

## CrossEntropyLoss

Foi utilizada a função:

```python
CrossEntropyLoss
```

Essa é a função de perda mais utilizada em problemas de classificação multiclasse.

Ela compara a resposta produzida pela rede com a resposta correta do dataset.

Quanto maior o erro, maiores serão os ajustes realizados nos pesos da rede.

---

## Adam Optimizer

Como algoritmo de otimização foi utilizado o:

```python
Adam
```

com taxa de aprendizado:

```python
learning_rate = 0.001
```

O Adam ajusta automaticamente os pesos da rede de forma eficiente, acelerando a convergência do treinamento.

---

# 4. Loop de treinamento

O treinamento foi realizado durante:

```text
10 épocas
```

Em cada época, a rede percorre todas as imagens do conjunto de treinamento.

Para cada lote de imagens são executadas as seguintes etapas:

1. Limpeza dos gradientes anteriores (`optimizer.zero_grad()`).
2. Propagação das imagens pela rede.
3. Geração das previsões.
4. Cálculo do erro utilizando a CrossEntropyLoss.
5. Aplicação do algoritmo de Backpropagation.
6. Atualização dos pesos utilizando o Adam.

Esse processo é repetido até que todas as imagens sejam utilizadas.

Ao final das dez épocas, a rede já aprendeu padrões suficientes para classificar novas imagens.

---

# 5. Salvamento do modelo

Após o treinamento, o script salva automaticamente os pesos aprendidos em:

```text
models/
└── tomato_cnn.pth
```

Esse arquivo contém todo o conhecimento adquirido pela rede durante o treinamento.

Dessa forma, não é necessário treinar novamente o modelo sempre que uma nova imagem precisar ser classificada.

Basta carregar esse arquivo para reutilizar a CNN treinada.

---

# 📊 Avaliação do Modelo

Depois do treinamento foi necessário verificar se a rede realmente havia aprendido.

Para isso foi desenvolvido o script:

```text
src/evaluate.py
```

Enquanto o script anterior treina a rede, este é responsável por medir seu desempenho utilizando imagens que não participaram do treinamento.

Essa separação é extremamente importante para avaliar a capacidade de generalização do modelo.

---

# 📄 Script: `src/evaluate.py`

O processo de avaliação ocorre em quatro etapas.

## 1. Preparação das imagens

As imagens do conjunto de validação recebem exatamente o mesmo pré-processamento utilizado durante o treinamento.

Isso garante que a rede receba dados no mesmo formato aprendido anteriormente.

---

## 2. Reconstrução da arquitetura

O script recria exatamente a mesma arquitetura utilizada durante o treinamento.

Essa etapa é necessária para que os pesos salvos possam ser carregados corretamente.

---

## 3. Carregamento do modelo treinado

Os pesos são carregados a partir do arquivo:

```text
models/tomato_cnn.pth
```

Logo em seguida, o modelo é colocado em modo de avaliação utilizando:

```python
model.eval()
```

Esse comando informa ao PyTorch que não haverá treinamento.

Assim, camadas específicas de treinamento (como Dropout, caso existissem) são desativadas automaticamente.

---

## 4. Cálculo da acurácia

Durante a avaliação também é utilizado:

```python
torch.no_grad()
```

Como nenhum aprendizado será realizado nessa etapa, o cálculo de gradientes é desnecessário.

Isso reduz o consumo de memória e torna a avaliação mais rápida.

Para cada imagem do conjunto de validação, o script:

- realiza a previsão;
- identifica a classe com maior probabilidade;
- compara essa previsão com o rótulo verdadeiro;
- contabiliza os acertos.

Ao final, a acurácia do modelo é calculada.

---

# 📈 Resultado obtido

A CNN apresentou aproximadamente:

```text
Acurácia ≈ 89%
```

Esse resultado demonstra que o modelo conseguiu aprender satisfatoriamente a tarefa proposta.

Durante os testes observou-se que a CNN apresentou excelente desempenho nas classes:

- Damaged
- Ripe
- Unripe

A principal dificuldade permaneceu na classe **Old**, cujas imagens apresentam características visuais muito semelhantes às da classe **Ripe**.

Esse comportamento já havia sido observado durante a análise exploratória do dataset.

---

# 🖼️ Testes com novas imagens

Após validar o modelo, foi realizado um teste utilizando imagens retiradas da internet.

Para isso foi desenvolvido o script:

```text
src/predict_image.py
```

---

# 📄 Script: `src/predict_image.py`

Esse script permite utilizar a CNN já treinada para classificar novas imagens individualmente.

Seu funcionamento é bastante simples.

1. Carrega o modelo salvo (`tomato_cnn.pth`);
2. Carrega uma imagem presente na pasta `imagens_teste`;
3. Realiza o mesmo pré-processamento utilizado durante o treinamento;
4. Executa a inferência da CNN;
5. Exibe a classe prevista juntamente com o nível de confiança.

Esse procedimento permite verificar o comportamento do modelo em imagens que nunca fizeram parte do dataset original.

---

# Resultados dos testes

Os testes mostraram que a CNN conseguiu classificar corretamente a maioria das imagens.

As classes **Damaged**, **Ripe** e **Unripe** apresentaram resultados bastante consistentes.

Entretanto, imagens de tomates classificados como **Old** continuaram sendo a principal fonte de erros.

Esse comportamento reforça a conclusão obtida durante a análise exploratória: a dificuldade está mais relacionada à qualidade e à semelhança visual das imagens do dataset do que propriamente à arquitetura da CNN.

De forma geral, os resultados foram considerados satisfatórios para os objetivos acadêmicos do projeto e serviram como base para a próxima etapa, que consistiu no treinamento do modelo de detecção de objetos utilizando o YOLO11.

# 🎯 Treinamento do Modelo de Detecção de Objetos (YOLO11)

Após concluir o treinamento da CNN, o próximo desafio foi desenvolver um modelo capaz de **localizar automaticamente os tomates** em imagens e vídeos.

Até esse momento do projeto, a CNN conseguia apenas classificar imagens contendo um único tomate. Entretanto, para que o sistema funcionasse em tempo real utilizando a webcam, era necessário que o computador fosse capaz de encontrar onde cada tomate estava na cena antes de classificá-lo.

Para resolver esse problema foi utilizado o **YOLO11 (You Only Look Once)**, um algoritmo para tarefas de **Detecção de Objetos (Object Detection)**.

Enquanto uma CNN de classificação recebe apenas uma imagem contendo um objeto e responde *"este tomate está maduro"*, o YOLO analisa toda a imagem simultaneamente, identifica onde estão os tomates e retorna a posição exata de cada um deles através de caixas delimitadoras (*Bounding Boxes*).

Esse modelo se tornou a primeira etapa do pipeline desenvolvido neste projeto.

---

# Por que utilizar o YOLO?

O YOLO foi escolhido principalmente por sua velocidade e eficiência.

Ao contrário de modelos tradicionais de detecção, que executam várias etapas independentes para localizar objetos, o YOLO realiza todo o processo em uma única passagem pela rede neural.

Essa característica permite sua utilização em aplicações que exigem processamento em tempo real, como:

- câmeras de segurança;
- veículos autônomos;
- robótica;
- inspeção industrial;
- sistemas de monitoramento agrícola.

No contexto deste projeto, essa velocidade era essencial, pois o objetivo final era detectar tomates utilizando a webcam sem que houvesse atrasos perceptíveis durante a execução.

Outra vantagem importante é que o YOLO fornece diretamente:

- localização do objeto;
- nível de confiança da detecção;
- classe detectada.

Neste projeto foi utilizada apenas uma classe durante o treinamento:

```text
Tomato
```

Toda a classificação do estado do tomate foi delegada à CNN desenvolvida anteriormente.

Essa divisão tornou o sistema mais organizado e facilitou o treinamento dos modelos.

---

# 📂 Preparação do Dataset

Diferentemente da CNN, que utiliza apenas imagens classificadas, o YOLO exige que cada objeto presente na imagem esteja anotado.

Isso significa que não basta informar que existe um tomate na fotografia.

Também é necessário informar exatamente **onde ele está**.

Para isso, cada tomate deve ser envolvido por um retângulo denominado **Bounding Box**.

Essas anotações são armazenadas em arquivos de texto contendo as coordenadas do objeto.

Como o dataset original não possuía essas informações, foi necessário criar um novo conjunto de dados específico para o treinamento do YOLO.

---

## Seleção das imagens

Devido ao tempo necessário para realizar as anotações manualmente, foram selecionadas aproximadamente **300 imagens** do dataset original.

A seleção foi realizada manualmente, buscando incluir tomates pertencentes às quatro categorias existentes:

- Damaged
- Old
- Ripe
- Unripe

Apesar da variedade entre as classes, posteriormente foi observado que muitas imagens escolhidas correspondiam a tomates italianos.

Essa característica acabou influenciando o desempenho do modelo, que passou a detectar esse tipo de tomate com maior facilidade.

---

## Organização inicial

Antes das anotações, todas as imagens foram reunidas em uma única pasta.

Somente após concluir todas as marcações elas foram divididas entre treinamento e validação.

Essa estratégia simplificou bastante o processo de anotação.

---

# 🖍️ Anotação das imagens utilizando o CVAT

Para realizar as anotações foi utilizada a ferramenta **CVAT (Computer Vision Annotation Tool)**.

O CVAT é uma plataforma amplamente utilizada em projetos profissionais de Visão Computacional para criação de datasets anotados.

Como a ferramenta funciona através do Docker, inicialmente foi necessário instalar o **Docker Desktop** e executar uma instância local do CVAT.

Depois disso, bastou importar as imagens e iniciar o processo de anotação.

---

## Bounding Boxes

Cada tomate foi marcado utilizando um retângulo envolvendo completamente o fruto.

Esses retângulos representam as **Bounding Boxes** utilizadas durante o treinamento do YOLO.

Após concluir todas as marcações, o projeto foi exportado utilizando o formato:

```text
YOLO Detection
```

Esse formato gera automaticamente:

- arquivos `.txt` contendo as coordenadas das Bounding Boxes;
- arquivo `data.yaml`;
- estrutura compatível com o treinamento do YOLO.

---

# ⚠️ Dificuldades Encontradas

A preparação do dataset foi, sem dúvida, a etapa mais trabalhosa de todo o projeto.

Diversos problemas surgiram antes que o treinamento pudesse finalmente começar.

---

## Problema 1 — Ferramentas incompatíveis

Inicialmente foram testadas outras ferramentas de anotação.

Entretanto, muitas delas estavam desatualizadas e apresentavam incompatibilidades com as versões das bibliotecas instaladas no computador.

Depois de diversas tentativas, optei pelo **CVAT**, que apresentou maior estabilidade e compatibilidade com o restante do projeto.

---

## Problema 2 — Tipo incorreto de anotação

Esse foi, provavelmente, o erro que mais consumiu tempo.

Nas primeiras horas de trabalho, utilizei **polígonos** para marcar os tomates.

Posteriormente descobri que o treinamento do YOLO Detection exige **Bounding Boxes**, ou seja, retângulos.

Como consequência, foi necessário refazer todas as anotações.

Apesar do retrabalho, a segunda etapa foi realizada muito mais rapidamente devido à experiência adquirida durante a primeira tentativa.

Esse erro reforçou a importância de compreender completamente os requisitos do modelo antes de iniciar a preparação do dataset.

---

## Problema 3 — Estrutura dos arquivos

Outro desafio foi compreender a estrutura esperada pelo YOLO.

Cada imagem precisava possuir um arquivo `.txt` correspondente contendo suas anotações.

Além disso:

- os nomes dos arquivos deveriam ser exatamente iguais;
- imagens e labels precisavam permanecer sincronizadas;
- o arquivo `data.yaml` deveria apontar corretamente para as pastas de treino e validação.

Pequenos erros nessa organização impediam completamente o treinamento.

---

## Problema 4 — Dataset dividido

Outro aprendizado importante foi perceber que **não basta dividir apenas as imagens**.

As anotações também precisam ser divididas exatamente da mesma forma.

Caso contrário, o YOLO não consegue localizar as labels correspondentes às imagens durante o treinamento.

---

# 📄 Script: `src/split_dataset.py`

Após finalizar todas as anotações utilizando o CVAT, foi desenvolvido o script:

```text
src/split_dataset.py
```

Esse script automatiza toda a organização do dataset utilizado pelo YOLO.

Seu objetivo é evitar erros durante a separação entre treinamento e validação.

---

## Funcionamento

O script realiza automaticamente as seguintes etapas:

1. lê todas as imagens anotadas;
2. localiza os respectivos arquivos `.txt`;
3. divide o conjunto em aproximadamente **80% para treinamento** e **20% para validação**;
4. copia simultaneamente imagens e labels para as pastas corretas;
5. mantém os nomes dos arquivos sincronizados.

Ao final da execução, o dataset fica organizado da seguinte forma:

```text
dataset_yolo/

├── images/
│   ├── train/
│   └── val/
│
├── labels/
│   ├── train/
│   └── val/
│
└── data.yaml
```

Essa estrutura é exatamente a esperada pela biblioteca **Ultralytics YOLO11**.

---

## Por que criar esse script?

Embora essa divisão pudesse ser realizada manualmente, isso aumentaria significativamente a chance de erros.

Como imagens e labels precisam permanecer sincronizadas, qualquer arquivo movido incorretamente impediria o treinamento.

Automatizar esse processo trouxe diversas vantagens:

- elimina erros humanos;
- garante que imagens e labels permaneçam correspondentes;
- reduz o tempo de preparação do dataset;
- permite recriar rapidamente o conjunto de treinamento sempre que necessário.

Ao final dessa etapa, o dataset estava completamente preparado para iniciar o treinamento do modelo YOLO11.

# 🚀 Treinamento do Modelo YOLO11

Com o dataset devidamente anotado e organizado, a etapa seguinte foi treinar o modelo responsável pela detecção dos tomates.

Todo o treinamento foi realizado através do script:

```text
src/train_yolo.py
```

Esse script utiliza a biblioteca **Ultralytics**, responsável por disponibilizar uma implementação moderna do algoritmo **YOLO11**, simplificando todo o processo de treinamento.

Durante essa etapa, o objetivo do modelo não é identificar se o tomate está maduro, verde ou estragado.

Sua única função é aprender a responder à seguinte pergunta:

> **"Existe um tomate nesta imagem? Se sim, onde ele está?"**

Ao final do treinamento, o modelo é capaz de localizar automaticamente os tomates em novas imagens e vídeos, retornando a posição exata de cada um deles através de **Bounding Boxes**.

---

# 📄 Script: `src/train_yolo.py`

O treinamento do YOLO pode ser dividido em algumas etapas principais.

## 1. Carregamento do modelo

Inicialmente, o script carrega um modelo pré-treinado disponibilizado pela biblioteca Ultralytics.

Esse procedimento é conhecido como **Transfer Learning**, técnica em que um modelo previamente treinado em um grande conjunto de dados é reutilizado e adaptado para uma nova tarefa.

Isso reduz significativamente o tempo necessário para o treinamento e melhora a capacidade de generalização do modelo.

---

## 2. Leitura do dataset

O script utiliza o arquivo:

```text
dataset_yolo/data.yaml
```

Esse arquivo informa ao YOLO:

- onde estão as imagens de treinamento;
- onde estão as imagens de validação;
- quantas classes existem;
- o nome de cada classe.

Como neste projeto o objetivo é detectar apenas tomates, foi utilizada apenas uma classe:

```text
Tomato
```

---

## 3. Processo de treinamento

Durante o treinamento, o YOLO executa repetidamente o seguinte processo:

1. Carrega um lote de imagens.
2. Analisa cada imagem.
3. Tenta localizar os tomates.
4. Compara suas previsões com as Bounding Boxes anotadas no CVAT.
5. Calcula o erro.
6. Ajusta automaticamente seus parâmetros internos.
7. Repete o processo durante todas as épocas de treinamento.

A cada nova época, espera-se que o modelo cometa menos erros e produza previsões cada vez mais precisas.

---

## 4. Salvamento dos resultados

Ao final do treinamento, a biblioteca Ultralytics cria automaticamente a pasta:

```text
runs/
```


Os principais arquivos são:

| Arquivo | Descrição |
|----------|-----------|
| `best.pt` | Modelo com melhor desempenho durante o treinamento. |
| `last.pt` | Modelo correspondente à última época treinada. |
| `results.png` | Evolução das métricas ao longo das épocas. |
| `confusion_matrix.png` | Matriz de confusão do treinamento. |
| `PR_curve.png` | Curva Precision × Recall. |
| `F1_curve.png` | Evolução da métrica F1-Score. |

Durante o restante do projeto foi utilizado o arquivo:

```text
best.pt
```

por representar o melhor modelo obtido durante o treinamento.

---

# ⚠️ Dificuldade Encontrada Durante o Treinamento

Embora a preparação do dataset já tivesse sido concluída, surgiu um novo problema antes do treinamento.

Ao executar o script, a biblioteca **Ultralytics** apresentou incompatibilidade com a versão do **Python 3.13** instalada no computador.

Inicialmente, imaginei que o erro estivesse relacionado ao dataset ou às configurações do YOLO, o que resultou em várias horas de testes e tentativas de solução.

Após investigar a documentação da biblioteca, identifiquei que a incompatibilidade estava relacionada à versão do Python.

A solução foi realizar o **downgrade para o Python 3.12**, versão totalmente compatível com a biblioteca utilizada.

Depois dessa alteração, o treinamento ocorreu normalmente.

Esse problema reforçou a importância de verificar previamente a compatibilidade entre bibliotecas e versões do ambiente de desenvolvimento.

---

# 📊 Resultados do Treinamento

O treinamento foi concluído em aproximadamente:

```text
1 hora
```

Ao final, o modelo apresentou métricas bastante satisfatórias para a quantidade de dados disponível.

| Métrica | Resultado Aproximado |
|---------|---------------------:|
| Precision | **≈ 80%** |
| Recall | **≈ 90%** |

Esses resultados indicam que o modelo conseguiu localizar corretamente a maior parte dos tomates presentes nas imagens.

Considerando que apenas aproximadamente **300 imagens anotadas** foram utilizadas, o desempenho foi considerado bastante satisfatório para os objetivos acadêmicos do projeto.

---

# 🖼️ Testes com Imagens

Depois do treinamento, foi desenvolvido o script:

```text
src/predict_yolo.py
```

Esse script permite testar o modelo utilizando imagens individuais.

O objetivo é verificar se o YOLO consegue localizar corretamente os tomates antes da integração com a CNN.

---

# 📄 Script: `src/predict_yolo.py`

Seu funcionamento é bastante simples.

1. Carrega o modelo `best.pt`;
2. Lê uma imagem presente na pasta `imagens_teste`;
3. Executa a detecção utilizando o YOLO;
4. Desenha as Bounding Boxes ao redor dos tomates encontrados;
5. Exibe o resultado na tela.

Essa etapa foi extremamente importante para validar o funcionamento do detector antes de utilizá-lo em tempo real.

---

# Resultados Obtidos

Os testes mostraram que o YOLO conseguiu localizar corretamente os tomates presentes na maioria das imagens utilizadas.

As Bounding Boxes apresentaram boa precisão e envolveram adequadamente os frutos detectados.

Entretanto, foi possível observar que o modelo apresentou desempenho superior em tomates do tipo italiano.

Esse comportamento ocorreu porque a seleção das aproximadamente 300 imagens utilizadas para treinamento foi realizada manualmente.

Mesmo havendo variedade entre as classes, a maior parte das imagens escolhidas continha tomates italianos.

Como consequência, o modelo aprendeu melhor as características desse tipo específico de tomate.

Esse resultado evidencia a importância da diversidade dos dados durante o treinamento de modelos de Inteligência Artificial.

---

# 🎥 Testes em Tempo Real

Após validar o funcionamento em imagens estáticas, o próximo passo foi verificar o desempenho utilizando vídeo ao vivo.

Para isso foi desenvolvido o script:

```text
src/webcam_yolo.py
```

---

# 📄 Script: `src/webcam_yolo.py`

Esse script utiliza o OpenCV para capturar continuamente imagens da webcam.

Cada frame é enviado ao modelo YOLO, que realiza a detecção em tempo real.

Quando um tomate é encontrado, o sistema desenha automaticamente uma Bounding Box ao redor do objeto.

Durante os testes foi utilizada uma confiança mínima de:

```python
conf = 0.60
```

Isso significa que apenas objetos detectados com pelo menos **60% de confiança** são exibidos.

Essa configuração reduz significativamente a quantidade de falsas detecções.

---

# Comportamento Observado

Os testes utilizando a webcam apresentaram resultados bastante satisfatórios.

Na maioria das situações, o modelo conseguiu localizar corretamente os tomates presentes na cena.

Entretanto, também foi observado que alguns objetos arredondados eram ocasionalmente identificados como tomates quando a confiança mínima era reduzida.

Esse comportamento era esperado, pois o detector foi treinado utilizando apenas cerca de 300 imagens anotadas.

Quanto maior a quantidade e a diversidade dos dados de treinamento, melhor tende a ser a capacidade de generalização do modelo.

---

# 💡 Considerações

O treinamento do YOLO11 representou uma das etapas mais desafiadoras de todo o projeto.

Além da necessidade de aprender um novo modelo de Deep Learning, foi necessário compreender todo o processo de preparação de um dataset para detecção de objetos, incluindo a anotação manual das imagens, organização das labels, configuração do arquivo `data.yaml` e estrutura esperada pela biblioteca Ultralytics.

Apesar das dificuldades encontradas, o modelo apresentou resultados bastante satisfatórios para um projeto acadêmico.

Os testes demonstraram que o YOLO foi capaz de localizar corretamente a maior parte dos tomates presentes nas imagens e vídeos, tornando possível sua integração com a CNN na etapa seguinte.

Essa integração permitiu construir um pipeline completo de Visão Computacional, no qual o YOLO identifica a posição dos tomates e a CNN determina automaticamente seu estado de conservação.

# 🤖 Integração dos Modelos: YOLO11 + CNN

Após o treinamento e a validação dos dois modelos de forma independente, a etapa final do projeto consistiu em integrá-los em um único sistema capaz de funcionar em tempo real utilizando a webcam.

Enquanto o **YOLO11** foi treinado para localizar tomates na imagem, a **CNN** foi treinada para identificar o estado de conservação de cada fruto.

Dessa forma, cada modelo ficou responsável por resolver apenas um problema específico:

- **YOLO11:** detectar onde estão os tomates na imagem.
- **CNN:** classificar cada tomate detectado como **Damaged**, **Old**, **Ripe** ou **Unripe**.

Essa arquitetura é bastante utilizada em aplicações reais de Visão Computacional, pois permite combinar modelos especializados em diferentes tarefas, tornando o sistema mais modular e facilitando futuras melhorias.

---

# 📄 Script principal: `src/webcam_yolo_cnn.py`

Toda a integração entre os modelos foi implementada no script:

```text
src/webcam_yolo_cnn.py
```

Esse é o script responsável por executar todo o pipeline de Visão Computacional em tempo real utilizando a webcam.

Seu funcionamento pode ser dividido em quatro etapas principais.

---

# 1. Inicialização dos modelos

A primeira etapa consiste em carregar os dois modelos previamente treinados.

## YOLO11

O script carrega automaticamente o arquivo:

```text
runs/detect/.../weights/best.pt
```

Esse arquivo contém os pesos aprendidos durante o treinamento do modelo de detecção.

Também foi definido um limite mínimo de confiança (**confidence threshold**) igual a:

```python
conf = 0.60
```

Isso significa que apenas detecções com pelo menos **60% de confiança** são consideradas válidas.

Essa configuração reduz significativamente a quantidade de falsos positivos.

---

## CNN

Logo após carregar o YOLO, o script instancia a arquitetura da Rede Neural Convolucional desenvolvida neste projeto.

Em seguida, são carregados os pesos salvos anteriormente:

```text
models/tomato_cnn.pth
```

A CNN é colocada em modo de avaliação utilizando:

```python
model.eval()
```

Como não haverá treinamento durante a execução da webcam, também são desabilitados os cálculos de gradientes do PyTorch, reduzindo o consumo de memória e melhorando o desempenho.

Por fim, são definidas as quatro classes que a CNN é capaz de identificar:

- Damaged
- Old
- Ripe
- Unripe

---

# 2. Captura de vídeo

Após a inicialização dos modelos, o OpenCV abre a webcam utilizando:

```python
cv2.VideoCapture(0)
```

A partir desse momento inicia-se um loop contínuo.

Cada frame capturado pela câmera é enviado para o YOLO11.

O modelo analisa toda a imagem e retorna:

- posição dos tomates encontrados;
- coordenadas das caixas delimitadoras (*Bounding Boxes*);
- nível de confiança da detecção.

---

# 3. Recorte do tomate e preparação da imagem

Esta é a etapa em que ocorre a integração entre os dois modelos.

Para cada tomate detectado pelo YOLO11, o sistema realiza automaticamente o recorte da região correspondente utilizando as coordenadas retornadas pelo detector.

Em seguida, essa imagem passa por um processo de pré-processamento antes de ser enviada para a CNN.

As seguintes etapas são executadas:

- conversão do espaço de cores de BGR para RGB;
- transformação da imagem em um objeto da biblioteca Pillow;
- redimensionamento para **256 × 256 pixels**;
- conversão para Tensor;
- adição da dimensão referente ao lote (*batch*).

Ao final desse processo, a imagem está exatamente no mesmo formato utilizado durante o treinamento da CNN.

---

# 4. Classificação e exibição dos resultados

Depois de preparada, a imagem é enviada para a Rede Neural Convolucional.

A CNN retorna quatro valores correspondentes às probabilidades das classes.

Em seguida, o script aplica a função **Softmax**, convertendo esses valores em probabilidades reais.

A classe com maior probabilidade é selecionada como resposta final.

Por último, utilizando o OpenCV, o sistema desenha sobre o vídeo:

- um retângulo indicando a posição do tomate;
- a classe prevista pela CNN;
- a confiança da CNN;
- a confiança do YOLO11.

Todo esse processo acontece continuamente para cada frame capturado pela webcam.

A execução é encerrada apenas quando a tecla **Q** é pressionada.

---

# 🔄 Pipeline Completo

O funcionamento completo do sistema pode ser representado pelo seguinte fluxo:

```text
Webcam
   │
   ▼
Captura de Frame
   │
   ▼
YOLO11
Detecta os tomates
   │
   ▼
Bounding Box
   │
   ▼
Recorte do tomate
   │
   ▼
Pré-processamento
   │
   ▼
CNN
Classifica o tomate
   │
   ▼
Softmax
   │
   ▼
Resultado Final
   │
   ▼
Exibição na Webcam
```

Esse pipeline permite que a detecção e a classificação ocorram de forma totalmente automática e em tempo real.

---

# 📊 Resultados Obtidos

Ao final do desenvolvimento, foram obtidos dois modelos independentes e posteriormente integrados.

## CNN

| Métrica | Resultado |
|---------|----------:|
| Classes | 4 |
| Imagens de treino | 6.501 |
| Épocas | 10 |
| Tempo de treinamento | Aproximadamente 1 hora |
| Acurácia | **≈ 89%** |

A CNN apresentou excelente desempenho para as classes:

- Damaged
- Ripe
- Unripe

A principal dificuldade ocorreu na classe **Old**, devido à grande semelhança visual com tomates maduros presentes no próprio dataset.

---

## YOLO11

| Métrica | Resultado |
|---------|----------:|
| Imagens anotadas | 300 |
| Tempo de treinamento | Aproximadamente 1 hora |
| Precisão (Precision) | ≈ 80% |
| Revocação (Recall) | ≈ 90% |

O modelo apresentou boa capacidade de localizar tomates em imagens e vídeos.

Mesmo utilizando apenas 300 imagens anotadas, conseguiu detectar corretamente a maior parte dos tomates durante os testes.

---

## Sistema Integrado

A integração entre YOLO11 e CNN apresentou funcionamento satisfatório.

Durante os testes em tempo real utilizando a webcam foi possível:

- detectar automaticamente tomates presentes na cena;
- localizar corretamente cada fruto;
- recortar apenas a região de interesse;
- classificar seu estado de conservação;
- apresentar todas essas informações diretamente sobre o vídeo.

O pipeline desenvolvido demonstrou que os dois modelos conseguem trabalhar em conjunto de forma eficiente.

---

# ⚠️ Dificuldades Encontradas

Este projeto apresentou diversos desafios durante seu desenvolvimento.

Cada dificuldade contribuiu para o aprendizado das etapas envolvidas na construção de modelos de Visão Computacional.

## 1. Qualidade do dataset

Embora o conjunto de dados possuísse milhares de imagens, algumas classes apresentavam exemplos muito semelhantes visualmente.

Em especial, tomates classificados como **Old** frequentemente se pareciam com tomates **Ripe**, dificultando o aprendizado da CNN.

Além disso, algumas imagens apresentavam baixa qualidade, iluminação inadequada ou pouca nitidez.

---

## 2. Anotação das imagens

O treinamento do YOLO exigiu que todas as imagens fossem anotadas manualmente.

Inicialmente, utilizei ferramentas de anotação incompatíveis com as versões das bibliotecas instaladas no computador.

Após diversas tentativas, optei pelo **CVAT**, que apresentou melhor estabilidade.

Outro erro cometido foi utilizar **polígonos** durante as primeiras anotações.

Posteriormente descobri que, para treinamento do YOLO Detection, era necessário utilizar **retângulos (Bounding Boxes)**.

Isso tornou necessário refazer todas as anotações.

---

## 3. Organização do dataset do YOLO

Outro desafio foi compreender corretamente a estrutura exigida pelo YOLO.

As imagens e seus respectivos arquivos de anotação precisavam possuir exatamente o mesmo nome.

Além disso, o arquivo `data.yaml` precisava apontar corretamente para as pastas de treinamento e validação.

Esses pequenos detalhes impediram o treinamento diversas vezes até que toda a estrutura fosse organizada corretamente.

---

## 4. Compatibilidade de versões

Durante o treinamento do YOLO encontrei incompatibilidade entre a biblioteca **Ultralytics** e o **Python 3.13**.

Após várias tentativas de solução, foi realizado o downgrade para o **Python 3.12**, versão na qual o treinamento ocorreu normalmente.

---

## 5. Quantidade limitada de imagens

Outra limitação importante foi a quantidade de imagens utilizadas para treinar o YOLO.

Enquanto a CNN foi treinada com mais de seis mil imagens, o detector recebeu apenas aproximadamente trezentas imagens anotadas.

Isso fez com que, ocasionalmente, o modelo confundisse objetos arredondados com tomates.

---

# 🚀 Melhorias Futuras

Embora os resultados obtidos tenham sido satisfatórios para fins acadêmicos, diversas melhorias podem ser implementadas futuramente.

Entre elas destacam-se:

- aumentar significativamente a quantidade de imagens anotadas para treinamento do YOLO;
- utilizar diferentes variedades de tomates para reduzir o viés do dataset;


Essas melhorias têm potencial para aumentar significativamente a precisão e a robustez do sistema.

---

# ✅ Conclusão

O objetivo principal deste projeto não foi desenvolver um sistema comercial de classificação de tomates, mas sim compreender e aplicar, na prática, todas as etapas envolvidas na construção de uma aplicação de Visão Computacional baseada em Deep Learning.

Durante o desenvolvimento foi possível trabalhar com todas as fases de um projeto de aprendizado de máquina:

- obtenção e análise dos dados;
- exploração do dataset;
- preparação das imagens;
- treinamento de uma Rede Neural Convolucional;
- treinamento de um modelo de detecção utilizando YOLO11;
- avaliação das métricas obtidas;
- integração de diferentes modelos em um único pipeline;
- execução do sistema em tempo real utilizando webcam.

Além dos resultados obtidos, o projeto proporcionou experiência prática com ferramentas amplamente utilizadas na área, como **PyTorch**, **OpenCV**, **Ultralytics YOLO11**, **CVAT** e **Docker**.

As dificuldades enfrentadas durante o desenvolvimento também contribuíram significativamente para o aprendizado, principalmente em aspectos relacionados à preparação dos dados, anotação de imagens, organização do dataset e compatibilidade entre bibliotecas.

De forma geral, o sistema desenvolvido atingiu os objetivos propostos e demonstrou, na prática, a aplicação dos principais conceitos de Visão Computacional e Aprendizado Profundo estudados ao longo da disciplina.

Mais do que os resultados numéricos obtidos, este projeto representou uma oportunidade de compreender todo o processo de desenvolvimento de modelos de Inteligência Artificial, desde a preparação dos dados até sua utilização em uma aplicação funcional executando em tempo real.
