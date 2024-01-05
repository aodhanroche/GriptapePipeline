from dotenv import load_dotenv

# Griptape
from griptape.structures import Pipeline
from griptape.tasks import PromptTask, ImageGenerationTask
from griptape.drivers import OpenAiDalleImageGenerationDriver
from griptape.engines import ImageGenerationEngine
from attr import define, field
from griptape.artifacts import TextArtifact
from griptape.tasks import BaseTextInputTask
from typing import Callable

load_dotenv()  # Load your environment


@define
class PythonTask(BaseTextInputTask):
    run_fn: Callable = field(kw_only=True)

    def run(self) -> TextArtifact:
        return self.run_fn()

# Variables
output_dir = "./images"

# Create the driver
image_driver = OpenAiDalleImageGenerationDriver(
    model="dall-e-3", api_type="open_ai", image_size="1024x1024"
)

# Create the engine
image_engine = ImageGenerationEngine(image_generation_driver=image_driver)

# Create the pipeline object
pipeline = Pipeline()

# Create tasks
python_task = PythonTask(
                run_fn= lambda python_task: print("This Python task ran")
            )

create_prompt_task = PromptTask(
    """
    Create a prompt for an Image Generation pipeline for the following topic: 
    {{ args[0] }}
    in the style of {{ style }}.
    """,
    context={"style": "a 1970s polaroid"},
    id="Create Prompt Task",
)

generate_image_task = ImageGenerationTask(
    "{{ parent_output }}",
    image_generation_engine=image_engine,
    output_dir=output_dir,
    id="Generate Image Task",
)

display_image_task = PromptTask(
    """
    Pretend to display the image to the user. 
    {{output_dir}}/{{ parent.output.name }}
    """,
    context={"output_dir": output_dir},
    id="Display Image Task",
)

# Add tasks to pipeline
pipeline.add_tasks(python_task, create_prompt_task, generate_image_task, display_image_task)

# Run the pipeline
pipeline.run("a cow")