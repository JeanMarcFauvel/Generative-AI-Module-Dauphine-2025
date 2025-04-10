document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');

    // Fonction pour ajouter un message au chat
    function addMessage(message, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
        messageDiv.textContent = message;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Fonction pour envoyer la question au serveur
    async function sendQuestion(question) {
        try {
            // Désactiver le bouton pendant l'envoi
            sendButton.disabled = true;
            sendButton.innerHTML = '<div class="loading"></div>';

            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: question })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            addMessage(data.response, false);
        } catch (error) {
            console.error('Error:', error);
            addMessage('Désolé, une erreur est survenue. Veuillez réessayer.', false);
        } finally {
            // Réactiver le bouton
            sendButton.disabled = false;
            sendButton.textContent = 'Envoyer';
        }
    }

    // Gestionnaire d'événement pour le bouton d'envoi
    sendButton.addEventListener('click', () => {
        const question = userInput.value.trim();
        if (question) {
            addMessage(question, true);
            sendQuestion(question);
            userInput.value = '';
        }
    });

    // Gestionnaire d'événement pour la touche Entrée
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendButton.click();
        }
    });
}); 