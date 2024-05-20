import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { useTransition, animated } from 'react-spring';
import 'bootstrap/dist/css/bootstrap.css';
import './equitos.css';


function Equitos() {
    const [messages, setMessages] = useState([]);
    const [inputText, setInputText] = useState('');
    const messagesEndRef = useRef(null);

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    const sendMessage = async () => {
        if (inputText.trim() === '') return;

        const newMessages = [...messages, { sender: 'user', message: inputText }];
        setMessages(newMessages);
        setInputText('');

        try {
            const response = await axios.post('http://127.0.0.1:5000/chat', { message: inputText });
            const botMessage = response.data.message;

            const updatedMessages = [...newMessages, { sender: 'bot', message: botMessage }];
            setMessages(updatedMessages);

            // Read out the bot's message
            speakMessage(botMessage);
        } catch (error) {
            console.error('Error:', error);
        }
    };

    const speakMessage = (message) => {
        const speech = new SpeechSynthesisUtterance(message);
        speech.lang='ta-IN'
        window.speechSynthesis.speak(speech);
    };

    const handleInputChange = (e) => {
        setInputText(e.target.value);
    };

    const messageTransitions = useTransition(messages, {
        from: { opacity: 0, transform: 'translateY(-20px)' },
        enter: { opacity: 1, transform: 'translateY(0)' },
        leave: { opacity: 0, transform: 'translateY(-20px)' },
    });

    return (
        <div className="container chat-container">
            <div className="chat-wrapper">
                <div className="chat-messages">
                    {messageTransitions((style, item, t, i) => (
                        <animated.div key={i} style={style} className={`message ${item.sender === 'user' ? 'user' : 'bot'}`}>
                            <p>{item.message}</p>
                        </animated.div>
                    ))}
                    <div ref={messagesEndRef} />
                </div>
                <div className="input-container">
                    <div className='form-group'>
                        <label style={{color:'white'}}>ENTER YOUR MESSAGE</label>
                        <input
                            type="text"
                            className="form-control"
                            value={inputText}
                            onChange={handleInputChange}
                            placeholder="Type a message..."
                            onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                        />
                        <button className="btn btn-primary" onClick={sendMessage}>Send</button>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default Equitos;
