# makefile 7ashak

ENTRY	=	langgraph_assignment.py

run:
	@echo "normal mode - agent is loading . . ."
	@uv run $(ENTRY)

# eval:
# 	@echo "eval mode - rag is loading . . ."
# 	@uv run $(ENTRY) --eval

push:
	@git add . && read -p "enter a commit message: " commit && git commit -m "$$commit" && git push