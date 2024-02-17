# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Copy the requirements file into the container at /usr/src/app
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
# (Assuming requirements.txt includes scrapy and scrapyd)
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir /etc/scrapyd
RUN mkdir -p /scrapyd/logs

# Make port 6800 available to the world outside this container
# (scrapyd default port)
EXPOSE 6800

COPY . .

# Run scrapyd when the container launches
CMD ["scrapyd"]