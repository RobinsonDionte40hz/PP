# Experience Memory System: The UBF's Long-Term Learning Mechanism

## The Problem Memory Solves

The Universal Behavioral Framework (UBF) updates consciousness coordinates when experiences occur, changing behavioral state immediately. But there's a limitation: **coordinates only capture current state, not past experiences.**

Without memory:
- A character who was betrayed recovers to normal frequency/coherence
- Meets the betrayer again
- Treats them like any other NPC
- Has learned nothing

**Memory provides the missing component: persistent learning from experience that influences future decisions.**

## Memory Structure: What Gets Stored

Every significant interaction creates a memory with 14 fields:

```rust
// From significant_memory.rs
pub struct Memory {
    pub id: String,                         // Unique identifier
    pub timestamp: u64,                     // When it happened
    pub significance: f64,                  // 0.0-1.0: How important?
    pub emotional_impact: f64,              // -1.0 to +1.0: How did it feel?
    pub interaction_type: InteractionType,  // Social/Combat/Economic/etc
    pub participants: Vec<String>,          // Who was involved?
    pub context: MemoryContext,             // Where/why/relevance scores
    pub decay_factor: f64,                  // 0.0-1.0: Recency weight
    pub interaction_id: String,             // Link to original interaction
    pub outcome: String,                    // "success"/"failure"/etc
    pub location: String,                   // Where it happened
    pub context_tags: Vec<String>,          // Tags like "betrayal", "victory"
    pub description: String,                // Human-readable description
}
```

### Memory Context: Why It Matters

```rust
pub struct MemoryContext {
    pub node_id: String,           // Location reference
    pub location: Option<String>,  // Descriptive location
    pub goal_relevance: f64,       // 0.0-1.0: Related to active goals?
    pub novelty_factor: f64,       // 0.0-1.0: First-time experience?
    pub social_importance: f64,    // 0.0-1.0: Important relationships?
    pub survival_relevance: f64,   // 0.0-1.0: Life-or-death situation?
    pub participants: Vec<String>, // Who else was there?
}
```

**These context scores directly feed into memory significance calculation.**

## Memory Formation: The Significance Filter

Not every interaction creates a memory. Only significant ones pass the **0.3 threshold**:

```javascript
// From SignificantMemoryService.js
calculateSignificance(event, context) {
    let significance = 0.0;
    
    // Component 1: Emotional Impact (0.0-0.4 weight)
    significance += Math.abs(event.emotionalImpact) * 0.4;
    
    // Component 2: Goal Relevance (0.0-0.3 weight)
    significance += context.goal_relevance * 0.3;
    
    // Component 3: Novelty Factor (0.0-0.2 weight)
    significance += context.novelty_factor * 0.2;
    
    // Component 4: Social Importance (0.0-0.1 weight)
    significance += context.social_importance * 0.1;
    
    // Component 5: Survival Relevance (0.0-0.1 weight)
    significance += context.survival_relevance * 0.1;
    
    // Total: 0.0-1.0
    return Math.min(1.0, significance);
}
```

**Result:** Maximum possible significance = 1.0 (0.4 + 0.3 + 0.2 + 0.1 + 0.1)

### Example Calculations

**Trivial Interaction (No Memory Formed):**
```
Event: "Buy bread from merchant"
- Emotional Impact: 0.05 → 0.05 × 0.4 = 0.02
- Goal Relevance: 0.1 → 0.1 × 0.3 = 0.03
- Novelty: 0.0 (routine) → 0.0
- Social Importance: 0.0 → 0.0
- Survival: 0.0 → 0.0
Total: 0.05 < 0.3 threshold → NOT STORED
```

**Significant Interaction (Memory Formed):**
```
Event: "Betrayed by trusted ally in battle"
- Emotional Impact: 0.9 → 0.9 × 0.4 = 0.36
- Goal Relevance: 0.8 (mission failure) → 0.8 × 0.3 = 0.24
- Novelty: 0.7 (first betrayal) → 0.7 × 0.2 = 0.14
- Social Importance: 0.9 (close ally) → 0.9 × 0.1 = 0.09
- Survival: 0.6 (near death) → 0.6 × 0.1 = 0.06
Total: 0.89 ≥ 0.3 threshold → STORED
```

## Memory Limits: Automatic Pruning

Characters can store **maximum 50 memories**:

```rust
pub const MAX_MEMORIES_PER_CHARACTER: usize = 50;
pub const MIN_SIGNIFICANCE_THRESHOLD: f64 = 0.3;
```

When the 51st memory is created:
1. Calculate weighted significance for all memories: `significance × decay_factor`
2. Sort memories by weighted significance (descending)
3. Keep top 50
4. Delete the rest

**Result:** Old, low-significance memories fade away. Recent, important memories persist.

## Memory Decay: Time Diminishes Impact

Memories don't stay fresh forever. The `decay_factor` represents how recent/relevant the memory remains:

```rust
// Simplified decay calculation
decay_factor = initial_decay * time_factor * relevance_factor;

// Example decay over time:
// Day 1:  decay = 1.0
// Week 1: decay = 0.9
// Month 1: decay = 0.7
// Year 1: decay = 0.3
```

**When used in decisions:**
- Recent memory (decay = 0.9): High influence
- Old memory (decay = 0.3): Low influence
- Ancient memory (decay = 0.1): Minimal influence

**But:** A highly significant memory (significance = 0.9) with some decay (0.5) still has weighted significance of 0.45, which can outweigh a recent but trivial memory (significance = 0.4, decay = 1.0 → weighted = 0.4).

## Memory Retrieval: Finding Relevant Experiences

When an agent makes a decision, the system retrieves relevant memories:

```rust
// From significant_memory.rs
pub fn retrieve_relevant_memories(
    memories: &[Memory],
    interaction_type: &InteractionType,
    max_count: usize,
) -> Vec<&Memory> {
    // Step 1: Filter by interaction type
    let mut relevant: Vec<&Memory> = memories
        .iter()
        .filter(|m| m.interaction_type == *interaction_type)
        .collect();
    
    // Step 2: Sort by weighted significance (significance × decay_factor)
    relevant.sort_by(|a, b| {
        let a_weight = a.significance * a.decay_factor;
        let b_weight = b.significance * b.decay_factor;
        b_weight.partial_cmp(&a_weight).unwrap()
    });
    
    // Step 3: Return top N memories
    relevant.into_iter().take(max_count).collect()
}
```

**Example:**
- Agent considers a Social interaction
- System retrieves all Social interaction memories
- Sorts by `significance × decay_factor` (most impactful recent experiences first)
- Returns top 10 memories
- Uses these to calculate memory influence

## Memory Influence: The Decision Modifier

This is where memory meets decision-making. Memories influence action selection through a **multiplier ranging from 0.8x to 1.5x**:

```rust
// From significant_memory.rs
pub fn calculate_memory_influence(
    memories: &[Memory],
    interaction_type: &InteractionType,
) -> Result<f64> {
    // Get relevant memories
    let relevant_memories = retrieve_relevant_memories(memories, interaction_type, 10);
    
    if relevant_memories.is_empty() {
        return Ok(1.0); // Neutral influence (no memory)
    }
    
    // Calculate weighted influence
    let total_influence: f64 = relevant_memories
        .iter()
        .map(|m| {
            // Emotional impact: -1.0 to +1.0
            let emotional_factor = m.emotional_impact;
            
            // Weight by significance and recency
            let significance_weight = m.significance * m.decay_factor;
            
            // Combined influence
            emotional_factor * significance_weight
        })
        .sum();
    
    let average_influence = total_influence / relevant_memories.len() as f64;
    
    // Map from [-1.0, 1.0] to [0.8, 1.5]
    // -1.0 → 0.8x (strong negative memory = avoid this action)
    //  0.0 → 1.0x (neutral memories = no bias)
    // +1.0 → 1.5x (strong positive memory = favor this action)
    let multiplier = 1.0 + (average_influence * 0.5);
    
    Ok(multiplier.clamp(0.8, 1.5))
}
```

### Memory Influence Examples

**Scenario 1: Positive Trading Memory**
```
Agent considers: "Trade with Merchant Guild"
Relevant memory: 
  - interaction_type: Economic
  - emotional_impact: +0.8 (great deal last time)
  - significance: 0.7
  - decay_factor: 0.9 (recent)
  
Calculation:
  weighted_influence = 0.8 × (0.7 × 0.9) = 0.8 × 0.63 = 0.504
  multiplier = 1.0 + (0.504 × 0.5) = 1.252
  
Result: Trade action weight multiplied by 1.252x
→ Agent more likely to trade with this merchant
```

**Scenario 2: Negative Betrayal Memory**
```
Agent considers: "Trust Lord Blackwood with secret"
Relevant memory:
  - interaction_type: Social
  - emotional_impact: -0.9 (betrayal)
  - significance: 0.9
  - decay_factor: 0.8 (somewhat recent)
  
Calculation:
  weighted_influence = -0.9 × (0.9 × 0.8) = -0.9 × 0.72 = -0.648
  multiplier = 1.0 + (-0.648 × 0.5) = 0.676
  → Clamped to minimum 0.8
  
Result: Trust action weight multiplied by 0.8x
→ Agent much less likely to trust this character
```

**Scenario 3: Mixed Memories**
```
Agent considers: "Enter combat"
Relevant memories:
  1. Victory: emotional +0.6, significance 0.7, decay 0.9
     influence = 0.6 × 0.63 = 0.378
  2. Defeat: emotional -0.7, significance 0.8, decay 0.8
     influence = -0.7 × 0.64 = -0.448
  3. Victory: emotional +0.5, significance 0.6, decay 0.7
     influence = 0.5 × 0.42 = 0.21
     
Average influence = (0.378 - 0.448 + 0.21) / 3 = 0.047
Multiplier = 1.0 + (0.047 × 0.5) = 1.024

Result: Combat action weight multiplied by 1.024x
→ Slight positive bias (victories slightly outweigh defeat)
```

## Integration with Decision Weighting

Memory influence is **Factor 5** in the 13-factor decision system:

```rust
// From interaction_weight.rs
pub fn calculate_interaction_weight(character, interaction) -> f64 {
    let mut weight = 1.0;
    
    // Factor 1: Goal alignment (+10.0 dominant boost)
    weight += calculate_goal_priority(character, interaction);
    
    // Factor 2: Critical needs (+8.0 for survival)
    weight += calculate_critical_needs(character, interaction);
    
    // Factor 3: Environmental suitability (0.1x-3.0x)
    weight *= calculate_environmental_suitability(character, interaction);
    
    // Factor 4: Personality influence (+2.0)
    weight += calculate_personality_influence(character, interaction);
    
    // Factor 5: MEMORY INFLUENCE (0.8x-1.5x multiplier)
    let memory_score = calculate_memory_influence(
        &character.memories,
        &interaction.interaction_type
    );
    weight *= memory_score; // 0.8-1.5x
    
    // Factor 6-13: Emotional state, needs, consciousness, etc.
    // ...
    
    return weight;
}
```

**Critical insight:** Memory influence is a **multiplier**, not an additive bonus. This means:
- Negative memories (0.8x) reduce weight significantly
- Positive memories (1.5x) boost weight substantially
- It affects the final decision more than most additive factors

## The Complete Feedback Loop

Here's how memory creates persistent learning:

### Cycle 1: Initial Experience

```
1. Character (freq: 7.5, coherence: 0.7) attempts "Trust Ally"
2. Behavioral state: social_drive = 0.6, risk_tolerance = 0.5
3. No relevant memories → memory_influence = 1.0x (neutral)
4. Action weight = 5.0 (before memory factor)
5. Action weight = 5.0 × 1.0 = 5.0 (after memory)
6. Character trusts ally → BETRAYED
7. Event outcome: "critical_failure"

Memory Formation:
  - significance = 0.89 (high)
  - emotional_impact = -0.9 (very negative)
  - interaction_type = Social
  - participants = ["ally_id"]
  - decay_factor = 1.0 (fresh)
  → Memory stored

Consciousness Update:
  - frequency: 7.5 → 7.0 (-0.5 for betrayal)
  - coherence: 0.7 → 0.6 (-0.1 for trauma)
  → Behavioral state regenerated
  → Character now has lower social_drive, lower risk_tolerance
```

### Cycle 2: Next Encounter (Same Character)

```
1. Character (freq: 7.0, coherence: 0.6) encounters same ally again
2. Behavioral state: social_drive = 0.4, risk_tolerance = 0.3 (lowered)
3. Considers "Trust Ally" action
4. Memory retrieval finds betrayal memory
5. Memory influence = 0.8x (strong negative)
6. Action weight = 4.0 (lower base due to behavioral state change)
7. Action weight = 4.0 × 0.8 = 3.2 (further reduced by memory)
8. Alternative "Be Cautious" has weight 6.5
9. Character chooses "Be Cautious" instead
→ Memory has changed future behavior
```

### Cycle 3: Recovery and Redemption

```
Time passes. Ally proves trustworthy again:

New positive interaction with same ally:
  - outcome: "success"
  - emotional_impact: +0.6
  - significance: 0.7
  → New memory stored

Now character has TWO memories with this ally:
  1. Betrayal (significance 0.89, emotional -0.9, decay 0.7)
  2. Redemption (significance 0.7, emotional +0.6, decay 1.0)

Average influence = [(-0.9 × 0.89 × 0.7) + (0.6 × 0.7 × 1.0)] / 2
                  = [-0.56 + 0.42] / 2
                  = -0.07

Memory influence = 1.0 + (-0.07 × 0.5) = 0.965

Result: Still slightly cautious (0.965x multiplier), but no longer strongly avoiding (was 0.8x).
→ Realistic gradual forgiveness through repeated positive experiences
```

## Memory Consolidation: Quantum-Inspired Optimization

For large memory sets, the system uses quantum-inspired algorithms for efficiency:

```rust
// From memory_management.rs
pub fn consolidate_memories_quantum(
    memories: &[Memory],
    consciousness_state: &ConsciousnessState,
) -> Vec<Memory> {
    // Step 1: Calculate quantum resonance patterns
    let weights: Vec<f64> = memories.iter().map(|m| {
        let recency_weight = m.decay_factor.powf(0.8);
        let significance_weight = m.significance.powf(1.2);
        let emotional_resonance = calculate_emotional_resonance(
            m.emotional_impact,
            consciousness_state
        );
        let coherence_amplifier = consciousness_state.emotional_coherence.powf(1.2);
        
        recency_weight * significance_weight * emotional_resonance * coherence_amplifier
    }).collect();
    
    // Step 2: Use quantum interference to remove redundant memories
    let filtered = quantum_interference_filter(memories, &weights);
    
    // Step 3: Consolidate similar memories into single representative
    let consolidated = weighted_memory_consolidation(&filtered, &weights);
    
    consolidated
}
```

**Purpose:** Instead of storing 10 similar "successful trade" memories, consolidate into 1-2 representative memories with averaged statistics. Saves space, maintains influence.

## Performance Characteristics

**Memory Operations (Rust/WASM):**
- Memory formation: ~2μs per event
- Significance calculation: ~1μs
- Memory retrieval (10 memories): ~5μs
- Influence calculation: ~3μs
- Memory pruning (50 → 50): ~50μs
- Full memory consolidation: ~200μs per 50 memories

**Memory Operations (JavaScript):**
- Memory formation: ~0.1ms per event
- Retrieval + influence: ~0.5ms
- Memory limits enforced automatically

**Storage:**
- 50 memories per character = ~8KB per character
- 10,000 characters = ~80MB for all memories
- Efficient compared to neural network weights

## Memory System Benefits

### 1. Persistent Learning Without Training

**Traditional ML:**
```
Train model on dataset → Deploy → Static behavior
To improve: Retrain entire model
```

**Memory System:**
```
Experience → Create memory → Influence future decisions
Continuous learning through operation
```

### 2. Explainable Decisions

You can inspect exactly which memories influenced a decision:

```javascript
// From BehavioralStateService.js
const memoryBreakdown = relevantMemories.map(memory => ({
    timestamp: memory.timestamp,
    significance: memory.significance,
    emotionalImpact: memory.emotionalImpact,
    outcome: memory.outcome,
    decayFactor: memory.decay_factor,
    influence: memory.emotional_impact * memory.significance * memory.decay_factor
}));

console.log(`Decision influenced by ${memoryBreakdown.length} memories`);
console.log(`Memory modifier: ${memoryModifier}x`);
```

**Result:** You can debug why an NPC chose an action by examining their memories.

### 3. Realistic Character Development

Characters develop "personalities" through experience:
- Repeatedly successful merchant → high confidence in trading (positive memories)
- Betrayed warrior → cautious with new allies (negative social memories)
- Veteran explorer → unafraid of danger (positive exploration memories)

**These aren't hand-coded personality traits. They emerge from accumulated experience.**

### 4. Dynamic Relationships

Relationship evolution happens naturally through memory:
- First meeting: No memories, neutral behavior
- Positive interactions: Positive memories accumulate, behavior becomes friendly
- One betrayal: Strong negative memory, relationship damaged
- Redemption arc: New positive memories slowly overcome negative ones

**No relationship variables to track. It's all in the memories.**

## Memory + UBF = Complete Adaptive System

The Universal Behavioral Framework provides:
- **Current state:** Frequency and coherence coordinates
- **Immediate response:** Behavioral state (energy, focus, mood, drives)
- **Adaptation:** Coordinate updates from outcomes

Memory adds:
- **Past context:** Accumulated experiences with significance weighting
- **Selective influence:** Only relevant memories affect current decisions
- **Gradual learning:** Repeated experiences shift behavior over time
- **Relationship memory:** Different behavior toward different entities

**Together:** A complete system where agents have both immediate emotional responses (UBF) and long-term learned behaviors (memory).

## Advanced Memory Patterns

### Pattern 1: Trauma Recovery

```
Major trauma → Large negative memory (significance 0.9, emotional -0.9)
→ Strong behavioral change (memory influence 0.8x for months)

Over time with therapy/positive experiences:
→ Decay factor reduces: 1.0 → 0.8 → 0.6 → 0.4
→ New positive memories accumulate
→ Average influence becomes less negative
→ Behavioral caution gradually fades

Realistic PTSD-like recovery trajectory
```

### Pattern 2: Skill Development

```
First attempt at blacksmithing:
- high novelty (0.8) → high significance
- likely failure → negative emotional impact
- memory influence: 0.85x (cautious)

After 10 attempts:
- novelty drops (0.2) → lower significance
- success rate improves → positive memories accumulate
- old failure memories decay
- memory influence: 1.2x (confident)

Natural skill acquisition curve through repeated practice
```

### Pattern 3: Trust Building

```
New ally relationship:
Day 1: No memories → neutral behavior (1.0x)

Week 1: 3 positive interactions
- Average emotional: +0.4
- Memory influence: 1.2x → slightly trusting

Month 1: 15 positive interactions
- Early memories decay, recent ones fresh
- Average emotional: +0.6
- Memory influence: 1.3x → trusting

Year 1: 100+ interactions consolidated
- Strong positive pattern
- Average emotional: +0.7
- Memory influence: 1.35x → deeply trusting

One betrayal: New strong negative memory
- Doesn't erase history but adds strong negative weight
- Average drops to +0.2
- Memory influence: 1.1x → cautiously positive
- Relationship damaged but not destroyed

Realistic trust development and betrayal impact
```

## Conclusion: Memory as Persistent Intelligence

The Memory System transforms the UBF from a reactive framework into a **learning system**:

**Without Memory:**
- Agents respond to current state only
- No learning from past mistakes
- Reset to baseline after each interaction
- Behavior is present-focused

**With Memory:**
- Agents accumulate experiential knowledge
- Past failures influence future caution
- Successful strategies become preferred
- Behavior reflects entire history

**The result:** Characters that genuinely learn, relationships that genuinely evolve, and behavior that genuinely adapts—all without neural networks, training data, or hand-coded rules.

Just: **experience → memory → influence → behavior change.**

---

## Summary Statistics

**Memory System Specifications:**
- **Storage limit:** 50 memories per character
- **Formation threshold:** 0.3 significance minimum
- **Influence range:** 0.8x to 1.5x decision weight modifier
- **Decay:** Time-based, reducing influence gradually
- **Retrieval:** Top 10 relevant memories per decision
- **Performance:** ~10μs per decision influence calculation (Rust)

**Integration with UBF:**
- **Factor 5** of 13 in decision weighting
- **Multiplier effect** (not additive)
- **Type-specific** (only relevant memories influence decisions)
- **Automatic consolidation** for efficiency
- **Explainable** (you can trace which memories influenced what)

**The Power:** Long-term behavioral adaptation through accumulated experience, making the UBF a complete adaptive intelligence framework.
