# Hand AI 

Here is the code used in the following article: TODO(zkbaum) add link to article.

## Structure

* `src/run_inference.py` - this is the main entry point for inference. It supports both zero and few shot inference with the ChatCompletions API.
* `src/retreival/run_assistants_v2_inference.py` - this is the main entry point for inference with file search. It uses the Assistants API instead of the ChatCompletions API. During the course of this project, OpenAI released the v2 assistants API, which is why there is some code for v1 and some for v2.
* `src/eval` - this contains all the code for analyzing the inference results and creating graphs. For example:
   * `src/eval/create_graph_with_ci.py` creates the file search result graph.
   * `src/eval/generate_p_values_anova.py` generates with p values with one-way anova. 


## Notices

* I am making the code public to show how the automation works. If you are interested in running this yourself, note the code will not work out-of-the-box because I did not make the dataset public (it is owned by the ASSH) and many paths are hard-coded to my machine. 
* This is a side project, meaning the code is not up to industry standards (e.g. no tests, limited documentation, etc). This does not reflect my code quality standards for professional software engineering projects.
* If you have any questions, feel free to open an issue.
