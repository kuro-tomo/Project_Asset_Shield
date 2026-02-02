# JÖRMUNGANDR V9.1 Strategy Configuration

# Allocation Strategy for Dual-Ledger
ALLOCATION = {
    "SAINT": 0.15,  # Public/Audit Ledger
    "GHOST": 0.85   # Private/Alpha Ledger
}

# Stealth & Obfuscation
STEALTH_SETTINGS = {
    "ENABLED": True,
    "DECOY_LOSS_PROBABILITY": 0.05
}

# Screening Criteria
# Mode: "INSTITUTIONAL" (Strict) or "BOOTSTRAP" (Lenient for small caps/growth)
SCREENING_MODE = "BOOTSTRAP" 

SCREENING_PARAMS = {
    "INSTITUTIONAL": {
        "Z_SAFE": 3.0,
        "F_TARGET": 8,
        "PEG_MAX": 1.0,
        "ALLOW_AGGRESSIVE": False
    },
    "BOOTSTRAP": {
        "Z_SAFE": 1.81, # Grey zone acceptable
        "F_TARGET": 6,  # 6/9 is decent
        "PEG_MAX": 2.0, # Growth allowed
        "ALLOW_AGGRESSIVE": True
    }
}
