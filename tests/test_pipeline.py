from pipeline import ExperimentPipeline


class TestPipeline:
    def test_pipeline(self):
        pipeline = ExperimentPipeline("configs/test_config.yaml")
        pipeline.run()
