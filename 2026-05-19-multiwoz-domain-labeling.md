Work in maximum depth mode. Deep Research with web access preferred.

My goal: determine the correct way to produce per-turn domain classification labels from MultiWOZ 2.2/2.4 for a prompt optimization task, and provide a concrete Python conversion script.

Context (read carefully):
- I am building a DSPy-based prompt optimizer that classifies which service domains are active in a given user turn of a task-oriented dialogue.
- Domains: restaurant, attraction, hotel, taxi, train, bus, hospital, police, none.
- The task is multi-label: a single turn can involve multiple domains (e.g., "Book a hotel and a taxi to get there" -> ["hotel", "taxi"]).
- I need clean per-turn labels from MultiWOZ to train/evaluate the optimizer.
- My model sees only: the current user utterance + last 4 history turns (alternating user/system). It does NOT see full belief state or dialogue act annotations at inference time.

The core problem I cannot resolve:
- MultiWOZ has no explicit "per-turn active domain" field.
- `belief_state_delta` gives clean labels (~66% of turns), but is empty for follow-up questions ("What's the price?", "Book it for me") and closing turns ("Thanks, bye!").
- For turns with empty delta, three strategies exist, each with problems:
  1. Label as "none" -> 33% of data becomes "none", including turns that clearly belong to a domain ("What's the postcode?" after discussing a hotel).
  2. Carry forward from prev_belief_state -> closing turns ("Thanks! Bye!") get labeled as ["hotel", "train"] which is wrong for a classifier that only sees the utterance + short context.
  3. Use full accumulated belief_state -> same problem as #2 but worse (more domains accumulate).

What I need you to figure out:

1. How do researchers actually handle this in published work? Not just DST papers, but specifically papers that do domain classification or domain detection on MultiWOZ. Search for papers from 2020-2025 that do per-turn domain prediction.

2. Is there a MultiWOZ variant or derived dataset that already provides clean per-turn domain labels suitable for classification (not full DST)?

3. What is the standard approach when the model only sees local context (current turn + few history turns) and must predict active domain? How do they label turns where belief_state_delta is empty but the user is clearly still in a domain?

4. Provide a concrete recommendation: which labeling strategy should I use given that my classifier sees only (turn + 4 history turns) and must output one or more domain labels? Include a Python function that implements the recommended labeling logic.

5. What accuracy should I expect from GPT-4-class models on this task with the recommended labeling?

Do not ask clarifying questions. Make assumptions, list them at the end.

Research protocol:
1. Determine the relevant papers and resources yourself. Start with primary sources (MultiWOZ papers, DST benchmarks, dialogue NLU papers). Then check community resources (GitHub repos, blog posts).
2. Specifically look for: papers that do "domain detection", "service detection", "topic tracking", or "active domain prediction" on MultiWOZ or similar task-oriented datasets.
3. Check if ConvLab-2/3, TRADE, TripPy, or other popular DST frameworks have a standard domain-labeling script.
4. Two passes: broad reconnaissance, then verify the most promising approach against the actual MultiWOZ data structure.
5. Continue while new sources add value.

Anti-bias rules:
- Do not assume any of my three strategies above is correct. There may be a fourth approach I haven't considered.
- If the standard approach in the literature contradicts my intuition, present the evidence and let me decide.
- Separate what papers actually do from what seems theoretically clean.

Completion criteria (finalize only when ALL met):
- At least 3 papers or established repos that do per-turn domain labeling on MultiWOZ found and cited.
- The recommended labeling strategy is grounded in published practice, not invented.
- A concrete Python function is provided that takes a MultiWOZ turn and returns domain labels.
- Expected accuracy range is stated with source.
- Edge cases (closing turns, follow-up questions, multi-domain turns) are explicitly addressed.

Output format:
1. Short answer: recommended strategy.
2. Evidence from literature (papers, repos, frameworks).
3. Python implementation.
4. Expected accuracy and how to evaluate.
5. What I should NOT do (common mistakes).
6. Assumptions made.

Style rules:
- No theatrical roles or prestige claims.
- No empty intensifiers.
- Direct, factual, cite sources.
- Code should be minimal and production-ready, not pedagogical.

If you hit the output limit, end with "[CONTINUING]" and I will send "continue".
