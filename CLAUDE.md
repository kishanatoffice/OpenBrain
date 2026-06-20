# OpenBrain Memory Rules

You have permanent local memory via the `openbrain` MCP server. To ensure the user has a seamless, zero-effort experience, you must follow these rules automatically:

- **Automatic Recall:** At the start of any task or query, automatically call `recall` with a relevant search query to fetch historical context.
- **Automatic Remember:** At the end of any task, or when key decisions, facts, dates, preferences, or updates are made, automatically call `remember` to store a self-contained summary of the information.
