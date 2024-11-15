from django.db import models


class UploadedDocument(models.Model):
    file = models.FileField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    title = models.CharField(max_length=255, blank=True)
    content = models.TextField(blank=True)
    language = models.CharField(max_length=10, default='en')
    average_sentiment = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"{self.title or self.file.name} - {self.uploaded_at}"


class ChatSession(models.Model):
    session_id = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_interaction = models.DateTimeField(auto_now=True)
    current_document = models.ForeignKey(
        UploadedDocument, null=True, blank=True, on_delete=models.SET_NULL)

    def __str__(self):
        return self.session_id


class ChatMessage(models.Model):
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE)
    content = models.TextField()
    is_user = models.BooleanField()
    sentiment_score = models.FloatField(null=True, blank=True)
    relevant_document = models.ForeignKey(
        UploadedDocument, null=True, blank=True, on_delete=models.SET_NULL)

    # Additional fields for sentiment and emotion analysis
    user_sentiment = models.CharField(max_length=100, null=True, blank=True)
    response_sentiment = models.FloatField(null=True, blank=True)
    user_emotions = models.JSONField(null=True, blank=True)
    response_emotions = models.JSONField(null=True, blank=True)

    # Add the timestamp field
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['timestamp']
