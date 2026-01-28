import great_expectations as gx
from typing import Tuple, List

'''
Data checks:
1. Make sure all columns are present (DONE)
2. Data types of each column must match (i.e. expected data type is number, string, etc.)
3. ExpectColumnValuessToBeInSet for categoricals
4. ExpectColumnValuessToBeBetween for numericals
'''

def validate_data(df) -> Tuple[bool, List[str]]: 
    # Load dataset into GX dataset

    context = gx.get_context() # Data Context defines the storage location for metadata

    data_source = context.data_sources.add_pandas("pandas")
    data_asset = data_source.add_dataframe_asset(name='pd dataframe asset')

    batch_definition = data_asset.add_batch_definition_whole_dataframe('batch definition')
    batch = batch_definition.get_batch(batch_parameters={'dataframe': df})

    # ===== SCHEMA VALIDATION - ESSENTIAL COLUMNS ===== #

    suite = gx.ExpectationSuite('bank_schema')

    # Column existence checks
    for col in ['age', 'job', 'marital', 'education', 'default', 'balance', 
                'housing', 'loan', 'contact', 'day', 'month', 'duration',
                'campaign', 'pdays', 'previous', 'poutcome']:
        suite.add_expectation(gx.expectations.ExpectColumnToExist(column=col))


    # ===== RUN VALIDATION SUITE ===== #
    results = batch.validate(suite)
    results_dict = results.to_json_dict()

    # ===== PROCESS RESULTS ===== #
    checks = results_dict.get('results')
    if checks is None:
        checks = [results_dict]
    
    failed_expectations: List[str] = []
    for r in checks:
        if not r.get('success', False):
            cfg = r.get('expectation_config', {}) or {}
            exp_type = cfg.get('expectation_type') or cfg.get('type') or 'unknown_expectation'
            failed_expectations.append(exp_type)
    
    total_checks = len(checks)
    passed_checks = sum(1 for r in checks if r.get('success', False))
    failed_checks = total_checks - passed_checks
    overall_success = bool(results_dict.get('success', False))

    if overall_success:
        print(f"Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"Failed expectations: {failed_expectations}")

    return overall_success, failed_expectations
