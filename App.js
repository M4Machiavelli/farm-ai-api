import React, { useEffect, useState } from 'react';
import { View, Text, Button } from 'react-native';
import SmsAndroid from 'react-native-get-sms-android';

const App = () => {
    const [waterLevel, setWaterLevel] = useState('Unknown');

    const fetchSMS = () => {
        const filter = {
            box: 'inbox',
            read: 0,
            address: '+1234567890', // SIM800 sender number
        };

        SmsAndroid.list(JSON.stringify(filter), (fail) => {
            console.log('Failed with error: ' + fail);
        }, (count, smsList) => {
            let messages = JSON.parse(smsList);
            if (messages.length > 0) {
                setWaterLevel(messages[0].body);
            }
        });
    };

    return (
        <View style={{ padding: 20 }}>
            <Text style={{ fontSize: 20, fontWeight: 'bold' }}>Water Level Status:</Text>
            <Text style={{ fontSize: 16, color: 'blue' }}>{waterLevel}</Text>
            <Button title="Check Water Level" onPress={fetchSMS} />
        </View>
    );
};

export default App;
