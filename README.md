# TASK 1
## 🔑 Design Choices
Used **list** as a Data Structure which can handle a million records <br>
Memory: 1M readings → store in list (fine, ~8 MB). 

_getAverage(k)_
- Maintaining a pre calculated sum for faster queries. 
- Maintain prefix[i] = sum of first i readings.
- So, Average of last k readings = prefix[𝑛]−prefix[𝑛−𝑘]/k
	​
_getMaxWindow(k)_
This is equivalent to finding the maximum subarray sum of length k.
With pre claculated sums, window sum for [i, i+k-1] = prefix[i+k] - prefix[i].

## ⚡ Performance
- addReading → O(1).
- getAverage(k) → O(1).
- getMaxWindow(k) → O(n) worst case.



# TASK 2
## 🧩 Async LangGraph Conversational Agent  

This project is a **lightweight conversational agent** built with [LangGraph](https://python.langchain.com/docs/langgraph/), [LangChain](https://www.langchain.com/), and OpenAI.  
The agent can detect user intent, route queries to the correct sub-agent, and maintain short-term memory across turns.  

## 🚀 Features  
- 🔍 **Intent Detection** → Classifies input as **factual** or **creative**  
- 📘 **Factual Agent** → Provides concise fact-based answers  
- 🎨 **Creative Agent** → Generates imaginative responses (captions, stories, etc.)  
- 🧠 **Short-term Memory** → Keeps track of the last 2–3 turns  
- ⚡ **Async Execution** → Uses non-blocking OpenAI API calls  
- 💬 **Ongoing Conversation** → Keeps chatting until you exit  


## 📦 Setup  

1. **Clone the repository**  
```bash
git clone https://github.com/ravi1173/Thrifty_AI.git
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create a .env file
```bash
OPENAI_API_KEY=your_openai_api_key
```
4. Run
```bash
python main.py
```


# TASK 3
Demonstrates binary classification using Logistic Regression on a synthetic dataset of real and fake data points.

1. Data Generation
- Real data (y=1): Generated using make_blobs with one cluster, 200 samples, and Gaussian spread (std=2).
- Fake data (y=0): Generated uniformly in the range [-6, 6] for both features.

2. Train-Test Split
- The dataset is split into 75% training and 25% testing.

3. Model Training
- A Logistic Regression classifier is trained on the training data.

4. Evaluation
- Predictions are made on the test set.

5. Model performance is evaluated using:
- Accuracy score
- Confusion matrix
- ROC Curve & AUC score
- ROC Curve Visualization
