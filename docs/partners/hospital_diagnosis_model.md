# DRAFT, not sent. Request to a hospital research group for a diagnosis model

For Arjun to send. Recipient: a hospital or academic medical-centre research group, chosen by Arjun. Replace each bracketed field before sending. Nothing has been sent. No patient data may enter the WhiteBox E&E repository under any circumstance (master prompt rule 0.1.8).

---

**Subject:** Research collaboration: testing whether a diagnosis model's reasoning is clinically grounded, under a data-use agreement

Dear Dr [name],

I lead WhiteBox E&E, an independent research programme. It is developing a method that recovers the internal reasoning structure of a trained model: which intermediate variables it computes, which of them it actually uses, and how they combine. The method then predicts where the model will fail. A diagnosis model is a natural test, because the clinical reasoning chain (findings, then syndrome, then diagnosis) is known. We would like to ask whether your group would collaborate on one model under a data-use agreement.

**What we are asking for**
- Access to one trained diagnosis model your group has developed or validated, such as an imaging classifier or a model over structured findings. Access can be to the weights, or to activations and interventions inside your secure environment.
- A held-out evaluation set with the labels you already use, and the subset where the model is known to fail, if one exists.
- A clinician's mapping of the intended reasoning chain, for the variables we should expect to find.

**What we would test (pre-registered before access)**
1. Are the model's diagnoses grounded in clinically meaningful intermediate variables, or in shortcut variables such as scanner, site or text markers?
2. Does our internal "memorization" verdict, which needs no held-out data, flag shortcut-reliant behaviour before held-out failure appears?

Every result is reported, including failures.

**Data protection**
- A formal data-use agreement and IRB or ethics approval on your terms. If required, all computation runs on your infrastructure, and nothing identifiable leaves it.
- We need model internals and aggregate statistics only. No patient-level data is copied to our systems or stored in our code repository.
- Your group reviews and co-authors any publication, as you prefer. Authorship is agreed in writing before work begins.

**What your group gets**
An audit of the model's reasoning against the clinical chain, a list of any shortcut variables found together with their causal evidence, and an early-warning measure for failure on new sites.

I would be glad to send the pre-registration draft and a short technical note, or to arrange a call. Thank you for considering this.

Kind regards,
Arjun Paradkar
WhiteBox E&E
[email] · [phone]
[Co-signers, if any: to be decided by Arjun]
