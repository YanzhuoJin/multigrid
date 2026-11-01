"""An audit must reject missing or inconsistent evidence, not infer success."""
import copy

from scripts.audit_submission import Audit, audit_submission, audit_training_rows


def rows_for_test():
    return [{"episode": str(i), "global_step": str(i * 100), "episode_length": "100",
             "agent_0_return": "1.0", "agent_1_return": "2.0", "agent_2_return": "0.0",
             "collective_return": "3.0", "agent_0_success": "1", "agent_1_success": "1",
             "agent_2_success": "0"} for i in range(1, 4)]


def test_absent_results_fail_with_named_missing_artifacts(tmp_path):
    result = audit_submission(tmp_path, evaluation_episodes=200)
    assert result["status"] == "fail"
    assert result["failed_checks"] >= 7
    failed = {check["check"] for check in result["checks"] if not check["passed"]}
    assert {"load_training_csv", "load_final_checkpoint", "load_training_summary",
            "evaluation_artifacts_integrity", "trajectory_media_integrity"} <= failed


def test_empty_summary_and_configuration_are_not_treated_as_success(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "summary.json").write_text("{}")
    result = audit_submission(tmp_path, evaluation_episodes=200)
    failed = {check["check"] for check in result["checks"] if not check["passed"]}
    assert {"assignment_training_configuration", "training_summary_complete"} <= failed


def test_complete_csv_arithmetic_passes_and_duplicate_episode_fails():
    rows = rows_for_test()
    audit = Audit()
    audit_training_rows(audit, rows, expected_episodes=3)
    assert all(check["passed"] for check in audit.checks)
    rows[1]["episode"] = "1"
    audit = Audit()
    audit_training_rows(audit, rows, expected_episodes=3)
    assert not next(check["passed"] for check in audit.checks
                    if check["check"] == "unique_contiguous_episode_indices")


def test_reward_tampering_and_nonfinite_values_fail():
    original = rows_for_test()
    for bad_value, expected_failed_check in (("7.0", "collective_equals_sum_of_agents"),
                                              ("nan", "finite_nonnegative_training_rewards")):
        rows = copy.deepcopy(original)
        rows[0]["collective_return"] = bad_value
        audit = Audit()
        audit_training_rows(audit, rows, expected_episodes=3)
        assert not next(check["passed"] for check in audit.checks
                        if check["check"] == expected_failed_check)
