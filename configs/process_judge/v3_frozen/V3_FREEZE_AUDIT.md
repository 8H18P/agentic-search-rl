# V3 Freeze Consistency Audit

Overall: PASS

- intent_contains_question: PASS
- intent_contains_gold: PASS
- intent_contains_prior_history: PASS
- intent_contains_reasoning: PASS
- intent_contains_query: PASS
- intent_excludes_current_result: PASS
- retrieval_contains_query: PASS
- retrieval_contains_current_result: PASS
- retrieval_excludes_original_question: PASS
- retrieval_excludes_gold: PASS
- retrieval_excludes_history: PASS
- retrieval_excludes_reasoning: PASS
- retrieval_builder_signature_is_query_result_only: PASS
- intent_schema_has_no_answer_tag: PASS
- retrieval_schema_has_no_answer_tag: PASS
- separate_intent_parser_present: PASS
- separate_retrieval_parser_present: PASS
- process_score_deterministic_and: PASS
- query_specific_positional_parser_present: PASS
- history_before_action_only: PASS
- raw_textual_action_not_serialized: PASS
- artifact_input_isolation_pass: PASS

No API, Retriever, calibration, held-out validation, or full Judge call was made by this audit.
