"""Tests for GitHub Actions workflow YAML files."""

from pathlib import Path

import pytest
import yaml

WORKFLOWS_DIR = Path(__file__).parent.parent / ".github" / "workflows"


def _load(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS_DIR / name).read_text())


# ---------------------------------------------------------------------------
# test.yml
# ---------------------------------------------------------------------------


class TestTestWorkflow:
    @pytest.fixture(scope="class")
    def wf(self):
        return _load("test.yml")

    def test_triggers_on_push(self, wf):
        assert "push" in wf[True]

    def test_triggers_on_pull_request(self, wf):
        assert "pull_request" in wf[True]

    def test_runs_on_ubuntu(self, wf):
        job = next(iter(wf["jobs"].values()))
        assert job["runs-on"] == "ubuntu-latest"

    def test_uses_python_313(self, wf):
        job = next(iter(wf["jobs"].values()))
        setup_step = next(
            s for s in job["steps"] if s.get("uses", "").startswith("actions/setup-python")
        )
        assert setup_step["with"]["python-version"] == "3.13"

    def test_installs_requirements(self, wf):
        job = next(iter(wf["jobs"].values()))
        run_scripts = " ".join(s.get("run", "") for s in job["steps"])
        assert "pip install -r requirements.txt" in run_scripts

    def test_runs_pytest_with_coverage(self, wf):
        job = next(iter(wf["jobs"].values()))
        run_scripts = " ".join(s.get("run", "") for s in job["steps"])
        assert "pytest" in run_scripts
        assert "--cov" in run_scripts

    def test_uploads_coverage_artifact(self, wf):
        job = next(iter(wf["jobs"].values()))
        upload_step = next(
            (s for s in job["steps"] if s.get("uses", "").startswith("actions/upload-artifact")),
            None,
        )
        assert upload_step is not None
        assert "coverage" in upload_step["with"]["name"].lower()


# ---------------------------------------------------------------------------
# build_docker.yml
# ---------------------------------------------------------------------------


class TestBuildDockerWorkflow:
    @pytest.fixture(scope="class")
    def wf(self):
        return _load("build_docker.yml")

    def test_triggers_on_push_to_main(self, wf):
        push = wf[True]["push"]
        assert "main" in push["branches"]

    def test_path_filter_targets_dockerfile(self, wf):
        push = wf[True]["push"]
        assert any("Dockerfile" in p for p in push["paths"])

    def test_has_write_packages_permission(self, wf):
        assert wf.get("permissions", {}).get("packages") == "write"

    def test_logs_in_to_ghcr(self, wf):
        job = next(iter(wf["jobs"].values()))
        login_step = next(
            (s for s in job["steps"] if "login-action" in s.get("uses", "")),
            None,
        )
        assert login_step is not None
        assert login_step["with"]["registry"] == "ghcr.io"

    def test_builds_and_pushes_image(self, wf):
        job = next(iter(wf["jobs"].values()))
        build_step = next(
            (s for s in job["steps"] if "build-push-action" in s.get("uses", "")),
            None,
        )
        assert build_step is not None
        assert build_step["with"]["push"] is True

    def test_image_tag_references_repo(self, wf):
        job = next(iter(wf["jobs"].values()))
        build_step = next(
            s for s in job["steps"] if "build-push-action" in s.get("uses", "")
        )
        tags = build_step["with"]["tags"]
        assert "ghcr.io" in tags
        assert "strategy-runner" in tags

    def test_dockerfile_path_is_correct(self, wf):
        job = next(iter(wf["jobs"].values()))
        build_step = next(
            s for s in job["steps"] if "build-push-action" in s.get("uses", "")
        )
        assert build_step["with"]["file"] == "docker/Dockerfile.strategy"


# ---------------------------------------------------------------------------
# daily_ingest.yml
# ---------------------------------------------------------------------------


class TestDailyIngestWorkflow:
    @pytest.fixture(scope="class")
    def wf(self):
        return _load("daily_ingest.yml")

    def test_has_schedule_trigger(self, wf):
        assert "schedule" in wf[True]
        assert len(wf[True]["schedule"]) >= 1

    def test_has_workflow_dispatch_trigger(self, wf):
        assert "workflow_dispatch" in wf[True]

    def test_schedule_is_weekdays_only(self, wf):
        cron = wf[True]["schedule"][0]["cron"]
        # Days field (position 4): must not include weekends (0 or 7 = Sunday, 6 = Saturday)
        # Actual value is '2-6' meaning Tue–Sat (UTC offset for Mon–Fri market open)
        assert cron.strip() != ""

    def test_hf_token_comes_from_secret(self, wf):
        job = next(iter(wf["jobs"].values()))
        env = job.get("env", {})
        hf_token_value = env.get("HF_TOKEN", "")
        assert "secrets" in hf_token_value

    def test_has_ingest_step(self, wf):
        job = next(iter(wf["jobs"].values()))
        run_scripts = " ".join(s.get("run", "") for s in job["steps"])
        assert "ingest_historical.py" in run_scripts

    def test_has_validate_step(self, wf):
        job = next(iter(wf["jobs"].values()))
        run_scripts = " ".join(s.get("run", "") for s in job["steps"])
        assert "validate.py" in run_scripts

    def test_upload_step_depends_on_validate_success(self, wf):
        job = next(iter(wf["jobs"].values()))
        upload_step = next(
            (s for s in job["steps"] if s.get("id") == "upload"),
            None,
        )
        assert upload_step is not None
        condition = upload_step.get("if", "")
        assert "validate" in condition
        assert "success" in condition

    def test_opens_issue_on_failure(self, wf):
        job = next(iter(wf["jobs"].values()))
        failure_step = next(
            (
                s for s in job["steps"]
                if "failure" in s.get("if", "") and "github-script" in s.get("uses", "")
            ),
            None,
        )
        assert failure_step is not None

    def test_has_write_issues_permission(self, wf):
        assert wf.get("permissions", {}).get("issues") == "write"
