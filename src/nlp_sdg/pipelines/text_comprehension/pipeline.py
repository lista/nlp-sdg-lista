"""
This is a boilerplate pipeline 'text_comprehension'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from nlp_sdg.pipelines.text_comprehension.nodes import summarize_text, question_and_answer



def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance =  pipeline(
        
        [
            node(
                func=summarize_text,
                inputs="model_input_data",
                outputs="summarize_text_data",
                name="summarize_text_node",
            ),
            node(
                func=question_and_answer,
                inputs="model_input_data",
                outputs="ques_and_ans_data",
                name="ques_and_ans_node",
            ),
        ]
    )
    text_comprehension = pipeline(
        pipe=pipeline_instance,
        inputs="model_input_data",
        namespace = "text_comprehension",
        outputs=["summarize_text_data", "ques_and_ans_data"]
    )
    return text_comprehension