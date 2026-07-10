# Final Report Content Guide

This note outlines the standard structure for your project report and serves as a reminder to follow key writing principles you've already been taught—such as:

- Clearly labelling all figures and graphs
- Using tables instead of long in-text lists or sentences of continuous text.
- Using introductions and summaries as appropriate at Chapter and Section level to structure the Report.
- Avoiding plagiarism and properly citing all sources

While much of this guidance applies broadly, **every project is unique**. Before you begin writing, consult with your supervisor to confirm the most appropriate structure and content for your specific report. **If your supervisor's advice differs from this guide, follow their direction**—they understand your project in context.

Your report must include **all mandatory sections specified in this Guide**. The internal structure generally follows a common pattern, but you may make small adjustments, provided that no essential content is omitted. For example:

- In some cases, the *Evaluation* chapter may be merged with *Results* or *Conclusions*.
- If you are not delivering usable software (e.g. in an analytical project), a *User Guide* may not be necessary.

Where appropriate, include **forward and backward references** to connect related material across sections. This helps the reader navigate your report and avoids unnecessary repetition.

## Formatting and Template Use

You must use the **official LaTeX report template**, modifying it by adding or removing Chapters as needed. The template includes customizable settings (e.g. for line spacing), which you may adjust if the default spacing does not suit your content.

Formatting rules and options:

- Use only the **approved font family** and **size** from the template
- Change the **doublespacing** flag to **false in main.tex** if your supervisor is OK with this to get more text on each page - useful if you want to keep figures or tables on the same page as their references.
- Set **edge_labels** flag to **false** in **main.tex** if you do not like them.
- Add separate **input** statements for each Chapter in **main.tex** as directed in the template. This is needed to avoid very long compilation times.
- Add your [statement of LLM usage (if relevant)](#) to the Declaration of Originality.
- In **ic_eee_thesis.cls** change `\LoadClass[a4paper,oneside]{book}` to `\LoadClass[a4paper,oneside]{book}` if you want to remove annoying blank pages after chapters. If you do this, you must switch edge tabs off.

### Template Usage Notes

- The template can be compiled locally with Luatex.
- The template is relatively slow for long documents
- **Watch out for the size of the generated PDF**. This will depend mostly on how you use images. If they have a resolution that is too high, or significant cropped areas, you may need PDF compression (eg the excellent and free [Adobe online compression service](#)) to get a long report PDF to yan acceptable size. Beware using third party PDF tools. Some are not safe.

The following sections provide a chapter-by-chapter guide to the typical contents of a project report.

---

## Front Matter

*This section was collapsed in the source document and its content is not included.*

## Introduction

*This section was collapsed in the source document and its content is not included.*

## Background

*This section was collapsed in the source document and its content is not included.*

## Requirements Capture

Projects with a deliverable that serves a specific function often have an initial phase in which expected use is investigated and a brief more detailed than the specification is constructed. This would include what is necessary, what is desirable, etc in the final deliverable. The results of requirements capture determine project objectives and are used to inform project evaluation. Requirements capture is important in all projects with real-world deliverables, and is often a significant amount of work in software projects.

Where requirements capture is less relevant (for example in an analytical 'research-style' project) this may be replaced by a detailed description of the project aims and objectives in the Introduction or the Background sections.

---

## Analysis & Design

If your project involved designing a system, begin with a clear **high-level overview** of the final design. This should help readers understand the overall structure and key components before diving into details.

In many cases, the final design will differ significantly from the original plan. If so:

- **Discuss the changes** and the reasons behind them.
- Highlight any **discoveries or challenges** that caused the project to shift direction or invalidate earlier work.
- Remember, you can gain credit not just for the end result, but for the **quality of the design process**, especially if it shows thoughtful adaptation and principled decision-making.

If your design was only **partially implemented**, be transparent:

- Clearly indicate which parts were completed and which were not.
- If certain components were not implemented due to unexpected challenges, limitations, or changing priorities, explain this—especially if those reasons reveal something interesting or insightful about the design process.

### Narrative and Structure

Your report is written at the **end of the project**, and should describe the **project as a whole**, not just the steps you took chronologically. Often, the clearest and most coherent description of your design is **not in the order it evolved**.

- Only describe the **evolution of ideas** where it adds value or insight.
- Otherwise, focus on a **structured, logical explanation** of the final design and its rationale.

### Emphasis on Engineering Thinking

Examiners are not only interested in what you built—they want to see **how you thought as an engineer**:

- Identify the **key design decisions** you faced.
- Present the **options you considered**.
- Justify the **choices you made**, including any trade-offs or constraints you had to balance.

Design decisions may be influenced by technical, practical, or even external factors. What matters is your ability to explain those decisions **rationally and critically**. Engineering is about making informed choices under constraints—your report should reflect that understanding.

---

## Implementation

For projects involving software, this section should provide a clear and focused overview of your implementation. Rather than including full code listings, describe **key components**, **important interfaces**, and **notable design decisions**. Emphasize the most **interesting, challenging, or surprising** aspects of your work.

You are not expected to cover everything—what matters is clarity and relevance. If you choose to omit parts of the implementation, state this explicitly and explain your reasoning.

### Including Code in the Report

Avoid copying large blocks of unedited code into the main report. Instead:

- Include **short, carefully chosen fragments** that illustrate important points—such as algorithmic flow, optimizations, use of data structures, or input formats for tools you've developed.
- **Edit code fragments** to remove unnecessary details and **annotate them** to help the reader understand their significance.
- Always ask yourself:
    - *What is this fragment demonstrating?*
    - *Does every line contribute to that message?*

If you cannot answer these questions, the code likely does not belong in the report.

### On the Use of Screenshots

Screenshots should be used **only when they add real value**. They are generally discouraged unless they:

- Show something that is **difficult to explain otherwise**, such as a failure mode, unexpected interaction, or visual result of a tool
- Clearly demonstrate a **specific concept**, such as a waveform, GUI interaction, or state machine behavior

Never use screenshots as page fillers, placeholders for proper diagrams, or as superficial "proof" that your tool ran. If you do include a screenshot, **crop, edit, and annotate it** to highlight the important content.

### Focus on Conceptual and Logical Design

When discussing your implementation:

- Begin with a **high-level overview** of your design—data flow, system architecture, key abstractions, etc.
- Use **diagrams and figures** (e.g. block diagrams, flowcharts, circuit hierarchies) to communicate structure and logic visually.
- Only then move into details, focusing on **parts that are technically interesting or central to your project's goals**.

For example, a block diagram of a digital circuit can efficiently convey architectural decisions, while small Verilog snippets may then illustrate specific, noteworthy logic components.

### Appendices and Code Repositories

- Full code listings should generally be included **only in an appendix**, and even then, only if they serve a clear documentation purpose (e.g. APIs).
- Your complete software should be submitted via a **cloud-based repository** such as GitHub. Make the repository public if possible; otherwise, ensure your supervisor has access.
- **No credit is awarded for report length**, and including printed code in place of meaningful explanation reflects poor judgement.

---

## Test Plan and Verification

This section should clearly describe **how your program or system was tested** and how its correct functioning was verified. A well-structured test plan demonstrates that your deliverable has been thoroughly evaluated and that its performance is understood.

- Include a concise summary of **test procedures**, **test cases**, and **results** in the main body of the report.
- If the test data is lengthy or repetitive but still relevant, place it in an **Appendix** for reference.
- Omit detailed test data **only** if it is not relevant, but always ensure the summary provided is accurate and complete.

*It is essential to be honest and precise. Clearly state what works, what doesn't, and how these conclusions were reached. Avoid presenting non-functional components as fully operational.*

Keep in mind that **examiners may attempt to compile, run, or test your deliverable themselves**, even after your report is submitted. Your report must accurately reflect the current state of your project. If additional work is completed after report submission, you can describe it during your **presentation or poster session**.

This section is especially important for **software and hardware projects**, and may be less applicable in primarily **analytical or theoretical** projects.

---

## Results

This section differs from the *Testing* chapter. While testing focuses on verifying functional correctness, performance analysis explores how well the algorithm, program, or hardware performs—either qualitatively or quantitatively.

If the deliverable has clearly measurable performance metrics, the *Testing* chapter will have established its correctness. This chapter, by contrast, assesses performance more deeply, examining aspects such as efficiency, accuracy, scalability, or responsiveness.

The following points outline typical content for this chapter. Not all may apply to every project:

- **Empirical Evaluation:** Explore relationships between key parameters and outcomes using graphs or tables to highlight trends and insights.
- **Comparison with Theoretical Expectations:** Where performance can be predicted by theory, compare observed results to expectations. Discuss any discrepancies and provide possible explanations.
- **Model Development:** Construct and test semi-empirical models to describe system behavior. Justify the modeling approach and assess its accuracy.
- **Experiment/Simulation Design:** Detail the types of experiments or simulations performed. Explain why certain experiments were chosen over others, and identify the key parameters involved. Discuss how variations in these parameters influenced the results.

**Note:** If your analysis includes numerous graphs or tables, consider placing them in an appendix. However, it's generally more helpful to present them alongside the relevant text for easier interpretation by the reader.

---

## Evaluation

### Critical Evaluation

This chapter (or a dedicated section within your Conclusions) is distinct from your *Results*. It should present a **critical assessment** of your work, specifically in relation to:

- Your original project objectives
- Existing analysis, algorithms, or products in the field

The goal is to demonstrate thoughtful engineering judgment, not just technical achievement. Examiners will look for evidence of your ability to critically reflect on your work and assess its significance.

#### Key Questions to Address:

- **To what extent were your original objectives achieved?**
- **If those objectives changed during the project, what was the rationale?**
- **What are the advantages and disadvantages of your approach compared to existing solutions?**
- **How does the scope of your project differ from related work?**

### Linking Back to Objectives and Requirements:

This section should compare your final outcomes to the initial requirements and objectives, typically defined in your Interim Report. If your goals evolved over the course of the project, you must explain these changes clearly and justify them within this chapter. Do **not** refer readers to the Interim Report—this section should be fully self-contained.

You should also reference (but not repeat) the earlier *Requirements Capture* section of your final report, summarizing the relevant points as needed to support your evaluation.

---

## Conclusions and Further Work

This chapter should provide a clear, honest, and thoughtful summary of your project's outcomes. It is your opportunity to highlight what you've achieved, acknowledge the limitations, and reflect on the significance of your work.

### Project Success and Achievements

- **How successful has the project been overall?**
- **What are the most important outcomes?** Summarise your key contributions and refer to relevant sections of the report for detailed evidence.
- **What aspects of your work are most worthwhile or impactful?** Be confident in identifying your successes.

At the same time, transparency is essential. Clearly acknowledge any limitations in your work, and identify tasks or ideas that remain unfinished. These can often be framed as valuable directions for **future work**, possibly to be pursued by another student in the future.

### Reflection and Design Decisions

- **What key design choices did you face, and why did you choose the path you did?**
- **What was the most challenging or inventive part of the project?**
    - Why was it difficult?
    - How did you address or overcome the challenge?
- **Did your work involve any novel approaches or discoveries?**
- **What did you learn—technically and professionally—during the course of the project?**

**Note:** The "most difficult" part isn't necessarily what took the most time—it's what required the most insight or problem-solving.

Your conclusions should be concise but meaningful, capturing the essence of the project and pointing the reader to other sections for full details where appropriate.

---

## Reflections

### Reflections Chapter

**This chapter is mandatory for BEng students only to pass your project, although it does not contribute to your final mark.** Its purpose is to demonstrate your understanding and application of the IET engineering competencies as they specifically relate to your project work.

#### To pass, you must:

- Clearly address **each required competency** in the context of your project within your Final Report.

#### You are strongly advised to:

- Include this chapter in the **draft report** submitted to your supervisor.
- Seek **specific feedback** to confirm that each competency is adequately addressed.

Please note:

- The IET competencies focus on **engineering soft skills**—such as communication, teamwork, and impact—not the technical content of your project.
- This chapter should therefore provide a **short, broad, non-technical discussion** that reflects on the professional and practical significance of your work.

If any part of this chapter is deemed inadequate:

- You will be required to revise your draft as directed by your supervisor.
- **Failure to adequately address the required competencies**—even if other aspects of the project are satisfactory—**may result in failing the project**.

While it is expected that you have developed these competencies through your degree or previous experience, it is **essential that you clearly articulate them here**.

#### Requirements:

- **MEng students: this Chapter is not required from 2025-6 onwards.**
- **BEng students:** Must complete and pass **six** required competency sections.

Please refer to the [official reflection guidelines](#) for detailed information on the required sections.

---

## User Guide

A software or hardware deliverable should have a User Guide focused on the intended user. Other projects may omit this. The User Guide may be a link to a public cloud-hosted (e.g. github) or in-app guide. Where this Chapter is short it may be presented as a section of the Implementation Chapter.
