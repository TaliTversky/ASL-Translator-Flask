from flask import Blueprint, render_template, request, flash, jsonify, redirect, url_for
from flask_login import login_required, current_user
from . import db
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

views = Blueprint('views', __name__)


@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    # if request.method == 'POST':
    #     destination = request.form.get('destination')
    #     if destination == 'learn':
    #         return redirect(url_for('learn.game'))
    #     elif destination == 'translate':
    #         return redirect(url_for('translate.live_translate'))
    
    return render_template("home.html", user=current_user)


@views.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        feedback = request.form['feedback']
        with open('feedback.txt', 'a') as f:
            f.write(f'Name: {name}, Email: {email}, Feedback: {feedback}\n')
        return render_template('thanks.html', user=current_user)
    else:
        return render_template('feedback.html', user=current_user)