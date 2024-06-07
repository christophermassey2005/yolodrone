from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain


# Initialize the Ollama model
llm = Ollama(model="llama3")

# Create a ConversationBufferMemory object
memory = ConversationBufferMemory()

# Create a ConversationChain with the Ollama model and memory
conversation = ConversationChain(llm=llm, memory=memory)

# SYSTEM prompt
prompt = "SYSTEM PROMPT: Your objective i"
result = conversation.predict(input=prompt)
print(result)
print()

# Second prompt
prompt = "What did I just tell you to do?"
result = conversation.predict(input=prompt)
print(result)