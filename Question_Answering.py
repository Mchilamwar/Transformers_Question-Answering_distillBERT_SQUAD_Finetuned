from transformers import pipeline
from transformers import AutoTokenizer,TFAutoModelForQuestionAnswering

# Importing Fully trained model on SQuad Dataset
tokenize=AutoTokenizer.from_pretrained("D:\\IVY Batches\\MY ML DL Projects\\Transformers Models\\Full trained squad QA")
model=TFAutoModelForQuestionAnswering.from_pretrained("D:\\IVY Batches\\MY ML DL Projects\\Transformers Models\\Full trained squad QA")

pipe=pipeline(task="question-answering",model=model,tokenizer=tokenize)


sample_context='''Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi\'s Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50.'''

sample_question='Who defeated National Football Conference Champion Carolina Panthers??'

def qna(context,question):

    if not context or not question:
        question=sample_question
        context=sample_context

    pred=pipe({'question':question,'context':context})

    score=str(round(float(pred.get('score')*100),2))+'%'
    answer=pred.get('answer')

    return (answer,score)

import gradio as gr
inp1=gr.inputs.Textbox(lines=10,placeholder="Sample Context:-\n"+sample_context,label="Input Context",optional=False)

inp2=gr.inputs.Textbox(lines=2,placeholder=sample_question,label="Input Question",optional=False)

out1=gr.outputs.Textbox(type='auto',label="Output Answer")
out2=gr.outputs.Textbox(type='auto',label="Score")

interface=gr.Interface(fn=qna,inputs=[inp1,inp2],outputs=[out1,out2],theme='dark-grass',title="Question Answering Distill BERT Fine tuned on SQUAD Dataset",live=False,allow_flagging='never')

interface.launch()

# print(qna())