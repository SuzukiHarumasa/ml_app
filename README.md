# 機械学習を利用したWeb APIの実装メモ



## 各フォルダの概要

### 01_make_api

予測モデルの作成から，Flaskの開発用アプリケーションサーバーで動作させるところまで[^1]



### 02_uwsgi

uWSGIサーバを設定するところまで[^2]



### 03_heroku

Herokuにデプロイする場合のファイル構成[^3]

（実際のデプロイ時は中にHeroku用のGitサブリポジトリを作る）



### 04_lambda

AWS lambda + API Gatewayでのデプロイ

- plain
  - Lambda関数だけを使うデプロイ[^5]
- with_chalice
  - chaliceでデプロイ[^4]
    （アプリ部分をFlaskからChaliceに書き換えることになるのであまりお手軽感は無かった）



### 05_fargate

AWS Fargate + ECS



### 06_EC2

AWS EC2 + ELB + AutoScaling





## ソースコードの詳細

[^1]: [機械学習モデルを動かすWeb APIを作ってみる(1)：APIの作成 - 盆暗の学習記録](https://nigimitama.hatenablog.jp/entry/2020/02/10/050000)
[^2]: [機械学習モデルを動かすWeb APIを作ってみる(2)：uWSGIの設定 - 盆暗の学習記録](https://nigimitama.hatenablog.jp/entry/2020/02/12/214018)
[^3]: [機械学習モデルを動かすWeb APIを作ってみる(3)：Herokuにデプロイ - 盆暗の学習記録](https://nigimitama.hatenablog.jp/entry/2020/02/17/000000)
[^4]: [機械学習モデルを動かすWeb APIを作ってみる(4)：chaliceでLambdaにデプロイ - 盆暗の学習記録](https://nigimitama.hatenablog.jp/entry/2020/02/25/000000)

