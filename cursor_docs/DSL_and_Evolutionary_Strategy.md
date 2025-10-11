# The SICA DSL and Evolutionary Strategy

## Part 1: The Foundation - An Abstract Language for Discovery

The core of this system is built on a fundamental principle: **the removal of human bias**. Traditional algorithmic trading is deeply rooted in human-designed concepts like Simple Moving Averages (SMA) or the Relative Strength Index (RSI). These concepts represent decades of human assumptions about market behavior.

Our Domain-Specific Language (DSL) rejects these assumptions by using a **tabula rasa** or "blank slate" approach.

### Abstract Symbolic Representation

Instead of human-meaningful terms, the DSL uses abstract, meaningless symbols:

```
ALPHA, BETA, GAMMA, DELTA, EPSILON, etc.
```

An initial strategy might look like this:
`IF ALPHA(10) > BETA(50) THEN BUY ELSE SELL`

The system does not know what `ALPHA` or `BETA` mean. They are simply placeholders for data inputs. This prevents the evolutionary process from being constrained by human trading folklore and allows for the discovery of genuinely novel patterns that humans may have never considered.

The meaning of these symbols and their combinations is not predefined; it is **discovered** through the unforgiving filter of market performance.

---

## Part 2: The Dual Engines of Improvement

The system's "genius" lies not in one single component, but in the symbiotic relationship between two distinct but complementary "engines" of discovery.

### Engine 1: The Evolutionary Engine (Broad Exploration)

This engine is the workhorse of discovery. It operates on the principle of natural selection through blind mutation.

-   **Mechanism**: Randomly mutates the DSL of surviving strategies (e.g., changing operators, parameters, or symbols).
-   **Cost**: Computationally "cheap." It can generate and test thousands of variations without incurring any LLM API costs.
-   **Function**: **Broad Exploration.** Its purpose is to relentlessly scan the vast landscape of possible strategies to find *any* signal, however faint, that demonstrates positive fitness. It provides the raw creative material and discovers the initial "footholds" of profitability.

### Engine 2: The Intelligence Engine (LLM-Guided Deep Search)

This engine provides focused, intelligent guidance. It is a scarce and "expensive" resource, applied selectively to accelerate progress.

-   **Mechanism**: Analyzes the structure, behavior, and performance of only the most successful "cells" discovered by the Evolutionary Engine.
-   **Cost**: Computationally "expensive," as it requires significant token context and powerful LLM reasoning.
-   **Function**: **Intelligent Exploitation and Acceleration.** Its purpose is to understand the *why* behind a successful strategy and then propose sophisticated, non-obvious improvements. It makes a few large, intelligent "jumps" in the search space, rather than many small, random steps.

---

## Part 3: The Symbiotic Loop in Action

Neither engine is sufficient on its own. Random evolution can be slow and get stuck on local maxima. Pure LLM generation can be uncreative and lack the spark of random discovery. The power of this system comes from their interaction, as detailed in the `EVOLUTION_WORKFLOW.md`.

The process is a powerful feedback loop:

1.  **Discover**: The Evolutionary Engine runs for many generations, finding a "promising" cell that has a clear fitness advantage through random mutation.
2.  **Analyze**: The Intelligence Engine is activated. The LLM receives the promising cell's genome, its performance data (phenotype), and its ancestry (lineage). It forms a hypothesis about why the strategy works.
3.  **Guide**: Based on its hypothesis, the LLM proposes a small number of high-quality, intelligent mutations. For example, it might identify a momentum component and suggest adding a volatility filter to improve it. These proposals are stored.
4.  **Exploit & Explore**: The system returns to the Evolutionary Engine. For most of its work, it continues with cheap, random mutations (exploration). But it now has a new capability: it can dedicate a fraction of its mutations (e.g., 20%) to testing the LLM's high-potential ideas (exploitation).
5.  **Validate**: The results of these guided mutations are tracked. When a child cell born from an LLM suggestion proves successful, it provides a positive feedback signal that the LLM's guidance was effective, creating a meta-learning loop.
6.  **Repeat**: The new, improved cell becomes the foundation for the next round of broad, random discovery, and the cycle repeats at a higher level of fitness.

This hybrid approach combines the best of both worlds: the relentless, unbiased creativity of evolution and the deep, contextual understanding of a large language model.

---

## Part 4: DSL V2 - The Language of Intelligence

This symbiotic loop would not be possible with a primitive language. The evolution from DSL V1 to DSL V2 was a critical step in enabling this strategy.

### The Failure of DSL V1

The initial version of the DSL was too simple. It could only compare raw indicator values, leading to nonsensical but accidentally profitable strategies. For example:

`IF EPSILON(20) <= DELTA(50) THEN HOLD ELSE BUY`

This compares `volume[t-20]` to `close[t-50]`. These are values of different units and magnitudes. While it might have found a random correlation, it represents no underlying market logic. An LLM analyzing this would be unable to form any meaningful hypothesis.

### DSL V2: Enabling Expressiveness and Analysis

DSL V2 introduced the features necessary for genuine strategic thinking by adding:

-   **Arithmetic Operations (`+`, `-`, `*`, `/`)**: Allows for the creation of ratios, spreads, and normalized values. This is the key that unlocked the ability to compare values meaningfully (e.g., `EPSILON(0) / AVG(EPSILON, 0, 20)` to compare current volume to its average).
-   **Aggregation Functions (`AVG`, `MAX`, `STD`)**: Allows for the creation of concepts like moving averages, support/resistance levels, and volatility filters.
-   **Logical Operators (`AND`, `OR`, `NOT`)**: Allows for the combination of multiple ideas into a single, cohesive strategy.

This richer language is essential for the Intelligence Engine. It gives the LLM a vocabulary to understand and describe complex market dynamics. An LLM can now look at a DSL V2 strategy and form a meaningful hypothesis (e.g., "This strategy is buying after a volume spike but only if the price has dipped below its recent average"), which in turn allows it to propose meaningful, intelligent improvements.
