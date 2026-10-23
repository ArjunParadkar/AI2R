# DRAFT, not sent. Request to quantitative firms for retired strategies

For Arjun to send. Recipients: firms with academic or research-partnership programmes, chosen by Arjun. Replace each bracketed field before sending. Nothing has been sent.

---

**Subject:** Research request: retired strategies and their decay curves, for a study of when models stop working

Dear [name],

I lead WhiteBox E&E, an independent research programme. It is building a method that reads a trained model's internal structure and predicts, before performance falls, whether the model has learned something durable or has memorized its training period. I am writing to ask whether [firm] would share one or more **retired** strategies, together with their realized decay curves, for a pre-registered study.

**What we are asking for**
- One or more strategies or models that [firm] has already retired. **We do not want, and will not accept, any live strategy.**
- Access to the model itself: weights, or a callable that we may query and intervene on. The training-period inputs, at whatever level of aggregation or anonymization you require.
- The realized out-of-sample performance after deployment, with the retirement date. This is the decay curve against which our prediction is scored.
- Delayed, anonymized or synthetic-feature versions are acceptable if they keep the time ordering.

**What we would do with it**
The study tests a single pre-registered prediction. It is registered with a timestamp before we see your data. The prediction is that our internal "depth" measure of the strategy falls before its live performance falls, while confidence and sharpness measures do not. If depth falls only together with performance, the test is recorded as a failure. We report every outcome, including failures.

**Protections for [firm]**
- An NDA and data-use agreement on your terms. Data and models can stay on your infrastructure if you prefer, and we would run our code there.
- No strategy logic, signal, feature or identifying detail is published. Results are reported as anonymized statistics, for example "strategy A: depth fell N days before performance".
- [Firm] reviews any manuscript before submission and can veto any detail it considers identifying.
- Acknowledgement or anonymity, as you prefer.

**Timeline**
We would start once an agreement is in place. A typical analysis takes [4–8] weeks of compute on our side.

I would be glad to send the pre-registration draft and a short technical note. Thank you for considering this.

Kind regards,
Arjun Paradkar
WhiteBox E&E
[email] · [phone]
[Co-signers, if any: to be decided by Arjun]
