# `product_data` — predicate assessment spec

The question this database answers:

> Given product X in country A, which products in countries B/C/D are close
> enough to serve as a regulatory precedent, and what evidence and pathway
> got them approved?

That gives two jobs: **match** (is it the same drug, same form, same use?) and
**precedent** (approved how, on what evidence?). Everything in `product_data`
serves one of those. Everything that serves neither is dropped.

Six top-level keys. Nothing else.

```
substance          what the drug is           → match
presentations      strength / form / route    → match
indications        what it is approved for    → match
approval           how it got approved        → precedent
pivotal_evidence   what evidence supported it → precedent
key_risks          headline safety profile    → precedent
```

---

## The contract

```json
{
  "_schema_version": "1",

  "substance": {
    "inn": "string — INN/normalised base name, the cross-country join key",
    "salt_form": "string|null — as formulated, when different from the base",
    "modality": "small_molecule | biologic | adc | vaccine | peptide",
    "target": "string|null — HER2, 5-HT3, alpha2-delta",
    "moa": "string|null — one line"
  },

  "presentations": [
    {
      "strength_value": 0, "strength_unit": "mg",
      "per_value": null, "per_unit": null,
      "form": "normalised dosage form",
      "route": "normalised route",
      "salt": { "substance": "", "value": 0, "unit": "mg" }
    }
  ],

  "indications": [
    {
      "condition": "string — normalised English",
      "population": "string|null — adults / paediatric / age bound",
      "line_of_therapy": "string|null — first-line / second-line / adjunctive",
      "biomarker": "string|null — HER2 IHC 2+ or 3+",
      "approval_date": "YYYY-MM-DD|null — per indication, not per product"
    }
  ],

  "approval": {
    "pathway": "innovator | generic | biosimilar | hybrid | similar",
    "registration_class": "string — the regulator's own class label",
    "conditional": false,
    "priority_review": false,
    "reference_product": "string|null"
  },

  "pivotal_evidence": [
    {
      "study_id": "string|null",
      "design": "string — single-arm phase II / randomised phase III",
      "n": 0,
      "endpoint": "ORR | PFS | OS | responder rate | non-inferiority",
      "value": 0, "unit": "% | months",
      "comparator": "string|null",
      "outcome": "met | not_met | null"
    }
  ],

  "key_risks": ["string — the handful that drive labelling and monitoring"]
}
```

`_schema_version` so parsers can migrate. Nothing here duplicates a column.

---

## Example 1 — Pregabalin Wockhardt 225 mg (MHRA)

```json
{
  "_schema_version": "1",
  "substance": {
    "inn": "Pregabalin",
    "salt_form": null,
    "modality": "small_molecule",
    "target": "Voltage-gated calcium channel alpha2-delta subunit",
    "moa": "Binds the alpha2-delta auxiliary subunit of voltage-gated calcium channels in the CNS"
  },
  "presentations": [
    { "strength_value": 225, "strength_unit": "mg", "form": "hard capsule", "route": "oral" }
  ],
  "indications": [
    { "condition": "Peripheral and central neuropathic pain", "population": "adults", "line_of_therapy": "monotherapy" },
    { "condition": "Partial seizures with or without secondary generalisation", "population": "adults", "line_of_therapy": "adjunctive" },
    { "condition": "Generalised anxiety disorder", "population": "adults", "line_of_therapy": "monotherapy" }
  ],
  "approval": {
    "pathway": "generic",
    "registration_class": "UK PL",
    "conditional": false,
    "priority_review": false,
    "reference_product": "Lyrica"
  },
  "pivotal_evidence": [
    { "design": "10 controlled trials up to 13 weeks", "endpoint": "50% pain reduction responder rate",
      "value": 35, "unit": "%", "comparator": "placebo 18%", "outcome": "met" },
    { "design": "6 controlled trials 4-6 weeks", "endpoint": "50% HAM-A improvement",
      "value": 52, "unit": "%", "comparator": "placebo 38%", "outcome": "met" },
    { "design": "Randomised 56-week monotherapy trial", "endpoint": "6-month seizure freedom, non-inferiority",
      "comparator": "lamotrigine", "outcome": "not_met" }
  ],
  "key_risks": [
    "Suicidal ideation and behaviour",
    "Respiratory depression",
    "Drug dependence and withdrawal syndrome",
    "Severe cutaneous adverse reactions (SJS/TEN)",
    "Major congenital malformations"
  ]
}
```

The failed lamotrigine non-inferiority is kept deliberately. A predicate whose
monotherapy claim failed is a materially different precedent from one that
succeeded, and that fact exists nowhere else in the row.

---

## Example 2 — ANSENTRON (ANVISA)

```json
{
  "_schema_version": "1",
  "substance": {
    "inn": "Ondansetron",
    "salt_form": "Ondansetron hydrochloride dihydrate",
    "modality": "small_molecule",
    "target": "5-HT3 receptor",
    "moa": "Potent, highly selective 5-HT3 receptor antagonist"
  },
  "presentations": [
    { "strength_value": 4, "strength_unit": "mg", "per_value": 2, "per_unit": "mL",
      "form": "solution for injection", "route": "intravenous, intramuscular",
      "salt": { "substance": "Ondansetron hydrochloride dihydrate", "value": 5, "unit": "mg" } },
    { "strength_value": 8, "strength_unit": "mg", "per_value": 4, "per_unit": "mL",
      "form": "solution for injection", "route": "intravenous, intramuscular",
      "salt": { "substance": "Ondansetron hydrochloride dihydrate", "value": 10, "unit": "mg" } }
  ],
  "indications": [
    { "condition": "Chemotherapy-induced nausea and vomiting", "population": "adults and children >= 6 months" },
    { "condition": "Radiotherapy-induced nausea and vomiting", "population": "adults and children >= 6 months" },
    { "condition": "Post-operative nausea and vomiting", "population": "adults and children >= 1 month" }
  ],
  "approval": {
    "pathway": "similar",
    "registration_class": "Medicamento Similar",
    "conditional": false,
    "priority_review": false,
    "reference_product": "Zofran"
  },
  "pivotal_evidence": [
    { "design": "Comparative trial vs high-dose metoclopramide (Marty 1990, NEJM)",
      "endpoint": "Control of cisplatin-induced emesis", "value": 75, "unit": "%",
      "comparator": "high-dose metoclopramide", "outcome": "met" }
  ],
  "key_risks": [
    "QT prolongation and torsades de pointes",
    "Myocardial ischaemia",
    "Serotonin syndrome",
    "Hypersensitivity including anaphylaxis"
  ]
}
```

`pathway: "similar"` is not `"generic"`. Brazil separates *genérico*, *similar*
and *referência*, and a similar is not interchangeable the way a genérico is.
Flattening it would make this look like a stronger equivalence precedent than
it is.

---

## Example 3 — 注射用维迪西妥单抗 / Disitamab Vedotin (NMPA)

```json
{
  "_schema_version": "1",
  "substance": {
    "inn": "Disitamab vedotin",
    "salt_form": null,
    "modality": "adc",
    "target": "HER2",
    "moa": "Recombinant humanised anti-HER2 monoclonal antibody conjugated to MMAE"
  },
  "presentations": [
    { "strength_value": 60, "strength_unit": "mg", "per_unit": "vial",
      "form": "powder for solution for infusion", "route": "intravenous" }
  ],
  "indications": [
    { "condition": "Urothelial carcinoma",
      "population": "adults, locally advanced or metastatic",
      "line_of_therapy": "after platinum-containing chemotherapy",
      "biomarker": "HER2 IHC 2+ or 3+",
      "approval_date": "2021-12-31" },
    { "condition": "Gastric cancer including gastro-oesophageal junction adenocarcinoma",
      "population": "adults, locally advanced or metastatic",
      "line_of_therapy": "after at least 2 prior systemic chemotherapies",
      "biomarker": "HER2 IHC 2+ or 3+",
      "approval_date": "2021-06-01" }
  ],
  "approval": {
    "pathway": "innovator",
    "registration_class": "Therapeutic Biological Product, Class 2.2",
    "conditional": true,
    "priority_review": true,
    "reference_product": null
  },
  "pivotal_evidence": [
    { "study_id": "RC48-C005", "design": "Open-label single-arm phase II", "n": 43,
      "endpoint": "ORR", "value": 51.2, "unit": "%", "outcome": "met" },
    { "study_id": "RC48-C009", "design": "Open-label single-arm phase II", "n": 64,
      "endpoint": "ORR", "value": 50.0, "unit": "%", "outcome": "met" },
    { "study_id": "RC48-C005 + C009 pooled", "design": "Pooled single-arm", "n": 107,
      "endpoint": "ORR", "value": 50.5, "unit": "%",
      "comparator": "historical chemotherapy ~10%, PD-1/PD-L1 15-25%", "outcome": "met" }
  ],
  "key_risks": [
    "Peripheral neurotoxicity",
    "Myelosuppression",
    "Abnormal liver function",
    "Gastrointestinal injury"
  ]
}
```

Both indications appear on one object because both sit under the same
registration `国药准字S20210017`. The 2021-12 filing **added** urothelial
carcinoma to a registration already approved for gastric cancer in 2021-06 —
so this array must be union-merged on upsert, never replaced.

This is also the strongest precedent record of the three: conditional approval
on single-arm ORR is exactly the pathway argument another applicant would cite.

---

## What was dropped, and why

| Dropped | Reason |
| ------- | ------ |
| Full ADR matrices (SOC × frequency, ~150 terms) | The flat `adverse_reactions` column answers "does this drug cause X". Frequency bands do not decide predicate equivalence. |
| Warning narratives | `key_risks` carries the headline profile. The prose is label content, not matching data. |
| Interaction lists | Not a matching axis. Nothing about predicate status turns on whether it interacts with phenytoin. |
| Detailed pharmacokinetics | Only matters for bioequivalence work, which is a different question from predicate assessment. Re-add as a `pharmacokinetics` key if BE becomes in scope. |
| Trial baseline demographics | Median age and metastasis rates do not select predicates. |
| Disease epidemiology | Background for a human reader, not data. |
| Risk management plans, post-market commitments | Regulatory follow-up, not equivalence. |
| Packaging, storage, shelf life, excipients | Supply-chain and formulation detail. **Caveat below.** |
| Review milestones, inspection findings | Process history. |
| `raw_sections` verbatim text | Useful as a re-parse escape hatch, but it is the single largest thing in the payload. Better stored on the raw record, which already holds the source document. |

**The one to think about: excipients.** They are out because they do not
identify the drug. But if you ever assess *pharmaceutical* equivalence for
generics — where a differing excipient can defeat an equivalence claim — they
come back. Adding `presentations[].excipients` later is cheap and non-breaking;
it is deliberately left out now rather than accidentally.

---

## Consequences for the columns

Two things from the worked examples are now load-bearing and belong on
`products`, not in JSON:

**`source_document_type`** — a review report cannot carry contraindications or
excipients. Without this, an empty array reads as "the drug has none" instead of
"this document type never had them". It changes how a NULL is interpreted, so it
is a column.

**`source_url TEXT` → `TEXT[]`** — the disitamab vedotin registration spans at
least three documents: two review reports and a label none of them contain.
`drug_predicate_raw_records.document_url` is already `TEXT[]` for this reason;
the products layer should match.
