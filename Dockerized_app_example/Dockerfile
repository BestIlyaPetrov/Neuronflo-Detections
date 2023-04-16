FROM python:3.8.2

WORKDIR /app

COPY requirements.txt .

# Install native libraries, required for numpy
# RUN apk --no-cache add musl-dev linux-headers g++

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "test_script.py" ]
