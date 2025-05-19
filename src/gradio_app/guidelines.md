# Entity Tag Guide

This document describes the annotation tags you will see in the NER / merged NER output. Each **entity** corresponds to a labeled span in the text.

| Entity                 | Meaning                                                                                                                                 |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **match\_named**       | Model and ground-truth agree on an explicit, uniquely named dataset span.                                                               |
| **actual\_named**      | A named dataset span present in the ground-truth but missed by the model.                                                               |
| **pred\_named**        | A named dataset span predicted by the model but not in the ground-truth.                                                                |
| **match\_unnamed**     | Model and ground-truth agree on a clearly described but unnamed dataset span.                                                           |
| **actual\_unnamed**    | An unnamed dataset span present in the ground-truth but missed by the model.                                                            |
| **pred\_unnamed**      | An unnamed dataset span predicted by the model but not in the ground-truth.                                                             |
| **match\_vague**       | Model and ground-truth agree on a vague dataset mention (lacking specific identifying details).                                         |
| **actual\_vague**      | A vague dataset mention present in the ground-truth but missed by the model.                                                            |
| **pred\_vague**        | A vague dataset mention predicted by the model but not in the ground-truth.                                                             |
| **source <> relation** | Relation-based annotation: highlights the dataset **target** span with a label combining source and relation (e.g. `RUV <> geography`). |

Use this guide when reviewing model predictions to quickly identify correct matches, false positives, and false negatives, as well as any extracted relations.
