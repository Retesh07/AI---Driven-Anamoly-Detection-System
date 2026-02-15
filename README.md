Input Frame
    ├── Pose Branch → Aggression Score
    ├── Weapon Branch → Weapon + Intent Score
    ├── Identity Branch → Known vs Unknown
    └── Tracking Module → Persistent ID over time

                ↓
        Behavioral Analyzer
        ├── Violence detection
        ├── Weaponized intent detection
        └── Loitering detection (unknown only)

                ↓
        Risk Scoring Engine
                ↓
        Edge Device Alert System
