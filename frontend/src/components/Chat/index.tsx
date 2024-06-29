//Modules
import gptAvatar from "@/assets/gpt-avatar.svg";
import warning from "@/assets/warning.svg";
import user from "@/assets/user.png";
import { useRef, useEffect } from "react";
import { useChat } from "@/store/chat";
import { useForm } from "react-hook-form";
import { useAutoAnimate } from "@formkit/auto-animate/react"
import { OpenAIApi, Configuration } from "openai";
import { useMutation } from "react-query";

//Components
import { Input } from "@/components/Input";
import { FiSend } from "react-icons/fi";
import {
    Avatar,
    IconButton,
    Spinner,
    Stack,
    Text
} from "@chakra-ui/react";
import ReactMarkdown from 'react-markdown'
import remarkGfm from "remark-gfm";
import { Instructions } from "../Layout/Instructions";
import { useAPI } from "@/store/api";

export interface ChatProps { };

interface ChatSchema {
    input: string
};

export const Chat = ({ ...props }: ChatProps) => {
    const { api } = useAPI();
    const {
        selectedChat,
        addMessage,
        addChat,
        editChat
    } = useChat();
    const selectedId = selectedChat?.id,
        selectedRole = selectedChat?.role;

    const hasSelectedChat = selectedChat && selectedChat?.content.length > 0;

    const {
        register,
        setValue,
        handleSubmit
    } = useForm<ChatSchema>();

    const [parentRef] = useAutoAnimate();

    const configuration = new Configuration({
        apiKey: api
    });

    const openAi = new OpenAIApi(configuration);

    const { mutate, isLoading } = useMutation({
        mutationKey: 'prompt',
        mutationFn: async (prompt: string) => await openAi.createChatCompletion({
            model: 'gpt-3.5-turbo',
            max_tokens: 256,
            messages: [{ role: 'user', content: prompt }]
        })
    });

    const handleAsk = async ({ input: prompt }: ChatSchema) => {
        const sendRequest = (selectedId: string) => {
            setValue("input", "");

            addMessage(selectedId, {
                emitter: "user",
                message: prompt
            });

            mutate(prompt, {
                onSuccess({ status, data }, variable) {
                    if (status === 200) {
                        const message = String(data.choices[0].message?.content);
                        addMessage(selectedId, {
                            emitter: "gpt",
                            message
                        });

                        if (selectedRole == "New chat" || selectedRole == undefined) {
                            editChat(selectedId, { role: variable });
                        };
                    }
                },
                onError(error) {
                    type Error = {
                        response: {
                            data: {
                                error: {
                                    code: "invalid_api_key" | string,
                                    message: string
                                };
                            },
                        },
                    };

                    const { response } = error as Error,
                        message = response.data.error.message;
                    addMessage(selectedId, {
                        emitter: "error",
                        message
                    });
                }
            });
        };

        if (selectedId) {
            if (prompt && !isLoading) {
                sendRequest(selectedId);
            };
        } else {
            addChat(sendRequest);
        };
    };

    const AlwaysScrollToBottom = () => {
        const elementRef = useRef<HTMLDivElement>(null);
        useEffect(() => {
            if (elementRef.current) {
                elementRef.current.scrollIntoView();
            }
        });
        return <div ref={elementRef} />;
    };

    return (
        <Stack
            width="full"
            height="full"
        >
            <Stack
                maxWidth="768px"
                width="full"
                marginX="auto"
                height="88%"
                overflow="auto"
            >
                <Stack
                    spacing={2}
                    padding={2}
                    ref={parentRef}
                    height="full"
                >
                    {(hasSelectedChat) ? (
                        selectedChat.content.map(({ emitter, message }, key) => {
                            const getAvatar = () => {
                                switch (emitter) {
                                    case "gpt":
                                        return gptAvatar;
                                    case "error":
                                        return warning;
                                    default:
                                        return user;
                                }
                            };

                            const getMessage = () => {
                                if (message.slice(0, 2) == "\n\n") {
                                    return message.slice(2, Infinity);
                                };

                                return message;
                            };

                            return (
                                <Stack
                                    key={key}
                                    direction="row"
                                    padding={4}
                                    rounded={8}
                                    backgroundColor={
                                        (emitter == 'gpt') ? ("blackAlpha.200") : ("transparent")
                                    }
                                    spacing={4}
                                >
                                    <Avatar
                                        name={emitter}
                                        mt={2}
                                        size="sm"
                                        src={getAvatar()}
                                    />
                                    <Text
                                        whiteSpace="pre-wrap"
                                        marginTop=".75em !important"
                                        overflow="hidden"
                                    >
                                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                            {getMessage()}
                                        </ReactMarkdown>
                                    </Text>
                                </Stack>
                            )
                        })
                    ) : (
                        <Instructions
                            onClick={(text) => setValue('input', text)}
                        />
                    )}
                    <AlwaysScrollToBottom />
                </Stack>
            </Stack>
            <Stack
                height="12%"
                padding={4}
                backgroundColor="blackAlpha.400"
                justifyContent="center"
                alignItems="center"
                overflow="hidden"
            >
                <Stack
                    maxWidth="768px"
                    width="100%"
                >
                    <Input
                        autoFocus={true}
                        variant="filled"
                        inputRightAddon={(
                            <IconButton
                                aria-label="send_button"
                                icon={(!isLoading) ? (<FiSend />) : (<Spinner />)}
                                backgroundColor="transparent"
                                onClick={handleSubmit(handleAsk)}
                            />
                        )}
                        {...register('input')}
                        onSubmit={console.log}
                        onKeyDown={(e) => {
                            if (e.key == "Enter") {
                                handleAsk({ input: e.currentTarget.value })
                            };
                        }}
                    />
                    {/* <Text
                        textAlign="center"
                        fontSize="sm"
                        opacity={.5}
                    >Free Research Preview. Our goal is to make AI systems more natural and safe to interact with. Your feedback will help us improve.</Text> */}
                    {/* {"TODO: We might want to say here something about how its not mean for education and to abide by course policy and honor code"} */}
                </Stack>
            </Stack>
        </Stack>
    );
};