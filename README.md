Template for creating and submitting MAT496 capstone project.

# Overview of MAT496

In this course, we have primarily learned Langgraph. This is helpful tool to build apps which can process unstructured `text`, find information we are looking for, and present the format we choose. Some specific topics we have covered are:

- Prompting
- Structured Output 
- Semantic Search
- Retreaval Augmented Generation (RAG)
- Tool calling LLMs & MCP
- Langgraph: State, Nodes, Graph

We also learned that Langsmith is a nice tool for debugging Langgraph codes.

------

# Capstone Project objective

The first purpose of the capstone project is to give a chance to revise all the major above listed topics. The second purpose of the capstone is to show your creativity. Think about all the problems which you can not have solved earlier, but are not possible to solve with the concepts learned in this course. For example, We can use LLM to analyse all kinds of news: sports news, financial news, political news. Another example, we can use LLMs to build a legal assistant. Pretty much anything which requires lots of reading, can be outsourced to LLMs. Let your imagination run free.


-------------------------

# Project report Template

## Title: PactScout

## Overview

A web app that lets you upload contracts (PDFs or Word files), automatically pulls out and standardizes all the clauses, and then highlights any legal or business risks — things like indemnity, termination, IP, confidentiality, or penalty clauses. It rates how risky each clause is, explains in simple terms why it’s risky, and gives you practical fixes: suggested wording, alternatives, or negotiation talking points.

Behind the scenes, it uses a RAG setup (your own document store + embeddings) so every explanation and suggestion is grounded in real examples. The whole process — from uploading the contract to extracting, analyzing, scoring, explaining, and finally presenting the results — is managed with LangGraph to keep the workflow reliable and modular.

## Reason for picking up this project

I picked this AI contract analyzer project because it connects nicely with what we learned in MAT496. Contracts are long and honestly very boring to read, and earlier I didn’t really know how to properly pull out the important points. But after learning things like prompting, structured output, semantic search, RAG, and how to make flows in LangGraph, I felt like I can actually build something that reads the contract for me. With LangGraph I can make small steps like “read text”, “find key info”, “check risky clauses”, and then give a simple final answer. This project helps me revise almost everything from the course while also making something that I would actually use in real life.

## Plan

I plan to excecute these steps to complete my project.

- Day 1: Foundation - Document Processing & Vector Store Setup
Goal: Set up document ingestion, embedding, and semantic search capabilities.[DONE]
- Day 2: Structured Output & Risk Analysis with Pydantic
Goal: Create structured output models and basic LLM-based risk analysis.[DONE]
- Day 3: RAG Integration & Tool Calling
Goal: Combine vector search with LLM analysis and add tool calling for enhanced retrieval.[DONE]
- Day 4: LangGraph Workflow - Complete System
Goal: Build the complete system using LangGraph with state management, nodes, and workflow.[DONE]

## Conclusion:

I had planned to achieve analysing a contract and let the LLM tell me what are the loopholes and risky clauses in a contract. I think I have achieved the conclusion satisfactorily. I can take any pdf or txt format contract and it will analyse it to tell me all about it, i feel like this can be really useful in a real world application so i feel quite satisfied with the end result of my capstone project for MAT496.

----------

# Added instructions:

- This is a `solo assignment`. Each of you will work alone. You are free to talk, discuss with chatgpt, but you are responsible for what you submit. Some students may be called for viva. You should be able to each and every line of work submitted by you.

- `commit` History maintenance.
  - Fork this respository and build on top of that.
  - For every step in your plan, there has to be a commit.
  - Change [TODO] to [DONE] in the plan, before you commit after that step. 
  - The commit history should show decent amount of work spread into minimum two dates. 
  - **All the commits done in one day will be rejected**. Even if you are capable of doing the whole thing in one day, refine it in two days.  
 
 - Deadline: Nov 30, Sunday 11:59 pm


# Grading: total 25 marks

- Coverage of most of topics in this class: 20
- Creativity: 5
  
