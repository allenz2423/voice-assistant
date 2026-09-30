# Adam Skills

Adam skills are Markdown files stored in this directory or in `~/.config/adam/skills/`.
User files take precedence over built-in files with the same skill ID.

`computer_use.md` and the detected desktop/window-manager guide load into Adam's
system prompt at startup. Adam can discover other skills with `list_skills` and
load their full guidance into the current conversation with `get_skill_context`.

Add a skill as a top-level `.md` file. Its filename, without the extension, is
its ID. For example, `~/.config/adam/skills/my_workflow.md` is loaded with
`get_skill_context(skill_name="my_workflow")`.
