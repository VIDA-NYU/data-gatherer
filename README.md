# data-gatherer

The Data Gatherer Tool is designed for automating the extraction of datasets from scientific articles webpages. It integrates LLMs, dynamic prompt management, and rule-based parsing, to facilitate data harmonization in biomedical research, and hopefully other domains.

When the Data Gatherer Tool locates a dataset, it will categorize access for that dataset in four categories: 
1. Easy download: The dataset consists of three or fewer files and can be downloaded without restriction.
2. Complex download: The dataset consists of four or more files and can be downloaded without restricton.
3. Application to access: Access to the dataset is restricted to those who complete an application and are approved. Application is handled by a centralized entity with clear procedures. 
4. Contact to access: The dataset may be available after the user contacts the originiating person or organization. Access may require application, but that application process is not clearly stated. 
