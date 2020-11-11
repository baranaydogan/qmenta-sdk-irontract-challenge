# See https://hub.docker.com/u/qmentasdk/ for more base images
FROM qmentasdk/minimal:latest

RUN apt-get update -y && \
    apt-get install -y wget

RUN wget https://github.com/dmritrekker/trekker/raw/master/binaries/trekker_linux_x64_v0.7
RUN chmod a+x trekker_linux_x64_v0.7

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN chmod a+x Miniconda3-latest-Linux-x86_64.sh
RUN sh Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda
RUN /miniconda/bin/conda install -c mrtrix3 mrtrix3


# A virtual x framebuffer is required to generate PDF files with pdfkit
RUN echo '#!/bin/bash\nxvfb-run -a --server-args="-screen 0, 1024x768x24" /usr/bin/wkhtmltopdf -q $*' > /usr/bin/wkhtmltopdf.sh && \
    chmod a+x /usr/bin/wkhtmltopdf.sh && \
    ln -s /usr/bin/wkhtmltopdf.sh /usr/local/bin/wkhtmltopdf

# Copy the source files (only this layer will have to be built after the first time)
COPY tool.py report_template.html qmenta_logo.png /root/

RUN pip install dipy





