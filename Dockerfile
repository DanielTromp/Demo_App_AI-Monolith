FROM danieltromp/py36-cv2
MAINTAINER "Daniël Tromp" <drpgmtromp@gmail.com>

ENV TERM=xterm \
    TZ='Europe/Amsterdam' \
    DEBIAN_FRONTEND=noninteractive

RUN echo $TZ > /etc/timezone

RUN apt-get update  --fix-missing \
    && apt-get -y upgrade \
    && apt-get --yes --no-install-recommends install apt-utils tzdata \
                                locales tzdata ca-certificates sudo \
    && dpkg-reconfigure -f noninteractive tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install gunicorn==19.7.1 sklearn Keras==2.2.4 matplotlib tensorflow==1.13.1

RUN git clone https://github.com/DanielTromp/Demo_App_AI-Monolith.git

WORKDIR /Demo_App_AI-Monolith/app/
RUN mkdir -p tmp
RUN wget -q https://www.dropbox.com/s/3pz96kng6hcupf5/age_model_weights.h5 -P models/
RUN wget -q https://www.dropbox.com/s/laum0pct5exj73r/gender_model_weights.h5 -P models/


EXPOSE 5555

CMD ["gunicorn", "--bind=0.0.0.0:5555", "app"]

# docker build --tag app-mono .
# docker stack deploy --compose-file docker-compose.yml app-mono
# docker run --detach -p 5555:5555 danieltromp/app-mono
