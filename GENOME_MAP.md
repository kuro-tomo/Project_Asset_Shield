graph TD
    subgraph "I. System Foundation"
        MC[MASTER_CONSTITUTION.md] -- Defines --> GP(Guiding Principles: 300億Cap, 低回転率, 生存バイアス排除, 自己増殖安全弁)
        GP -- Influences --> SA(System Architecture)
    end

    subgraph "II. Data & Preprocessing"
        JQA(J-Quants API) -- Provides Raw Data --> DI(Data Ingestion & Cache: data/jquants_cache.db)
        DI -- Processed by --> DE(Data Engineering: Pandas/Polars)
        DE -- Generates Features for --> SG(Signal Generation)
    end

    subgraph "III. Core Logic (src/shield)"
        SG -- Feeds --> AC(Adaptive Core: src/shield/adaptive_core.py)
        AC -- Adapts Parameters for --> BR(Brain AI: src/shield/brain.py)
        BR -- Generates Alpha Signals --> SG
        BR -- Learns from --> BF(Backtest Framework: src/shield/backtest_framework.py)
        BF -- Uses --> JQBP(JQuants Backtest Provider)
        AC -- Influences --> MM(Money Management: src/shield/money_management.py)
        MM -- Determines --> EC(Execution Core: src/shield/execution_core.py)
        EC -- Executes Orders via --> API(Broker API)
        SC(Screener: src/shield/screener.py) -- Provides Fundamental Analysis --> SG
        TR(Tracker: src/shield/tracker.py) -- Logs Events --> AD(Audit Database: logs/audit.db)
    end

    subgraph "IV. Self-Evolutionary Functions (src/shield/bio)"
        SR(Self-Repair: src/shield/bio/repair.py) -- Monitors Integrity --> System(Overall System)
        SE(Self-Evolution: src/shield/bio/evolution.py) -- Optimizes Parameters --> BR
        SP(Self-Replication: src/shield/bio/replication.py) -- Manages Instances --> EC
        BioCore(BioCore: src/shield/bio/core.py) -- Orchestrates --> SR & SE & SP
    end

    subgraph "V. Security & Governance"
        OSG(Operator Safety: docs/OPERATOR_SAFETY_GUIDE_JA.md) -- Details --> SIL(Silence Protocol: scripts/silence_protocol.py)
        SIL -- Protects IP & Operator --> BR & System
        DD(DD Cost Optimization: docs/DD_COST_AVOIDANCE_STRATEGY_JA.md) -- Strategy for --> PR(Platform Review)
    end

    subgraph "VI. Deployment & Reporting"
        INF(Infrastructure: infrastructure/) -- Deploys --> LIVE(Live Trading Environment)
        BF -- Generates --> R(Reports: output/)
        R -- Includes --> EP(Evidence Package: output/)
    end

    style MC fill:#f9f,stroke:#333,stroke-width:2px
    style GP fill:#bbf,stroke:#333,stroke-width:2px
