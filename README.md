# 機械学習を利用したWeb APIの実装メモ



## 各フォルダの概要

### 01_make_api

予測モデルの作成から，Flaskの開発用アプリケーションサーバーで動作させるところまで



### 02_uwsgi

uWSGIサーバを設定するところまで



### 03_heroku

Herokuにデプロイする場合のファイル構成

（実際のデプロイ時は中にHeroku用のGitサブリポジトリを作る）



### 04_lambda

AWS lambda + API Gatewayでのデプロイ

- plain
  - Lambda関数だけを使うデプロイ
- with_chalice
  - chaliceでデプロイ
    （アプリ部分をFlaskからChaliceに書き換えることになるのであまりお手軽感は無かった）



### 05_fargate

AWS Fargate + ECS



### 06_EC2

AWS EC2 + ELB + AutoScaling



